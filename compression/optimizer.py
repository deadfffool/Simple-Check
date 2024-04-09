# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright Microsoft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import warnings
from contextlib import contextmanager
import torch
from horovod.torch.mpi_ops import allreduce_async_, allgather_async, synchronize
from horovod.torch.mpi_ops import size, Average, rocm_built, global_process_set
from .utils import find_duplicates, get_comm, get_compressor, get_memory, get_config, check_not_compress, check_not_ef, get_check

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters,
                 comm_params = None,
                 backward_passes_per_step=1, op=Average,
                 gradient_predivide_factor=1.0,
                 sparse_as_dense=False,
                 process_set=global_process_set):
        super(self.__class__, self).__init__(params)

        self.named_parameters=named_parameters

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [(f'allreduce.noname.{i}.{j}', v)
                                for i, param_group in enumerate(self.param_groups)
                                for j, v in enumerate(param_group['params'])]
        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids = {id(v)
                         for param_group in self.param_groups
                         for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self.op = op
        self.gradient_predivide_factor = gradient_predivide_factor
        self.sparse_as_dense = sparse_as_dense
        self.process_set = process_set

        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._synchronized = False
        self._should_synchronize = True

        """
        initial communication information
        """
        self.world_size = size()
        self._comm_params = comm_params
        self.comm_mode = get_comm(self._comm_params)
        self.compressor = get_compressor(self._comm_params)
        self.memory = get_memory(self._comm_params)
        self.send_size_aresame = get_config(self._comm_params)
        self.check = get_check(self._comm_params)
        self.differential_dict = {}

        if self.process_set.included() and (size() > 1 or os.environ.get('HOROVOD_ELASTIC') == '1'):
            self._register_hooks()

    def load_state_dict(self, *args, **kwargs):
        self._handles = {}
        self._synchronized = False
        self._should_synchronize = True
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step
        super(self.__class__, self).load_state_dict(*args, **kwargs)

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    # def _allreduce_grad_async(self, p):
    #     if p.grad is None:
    #         p.grad = p.data.new(p.size()).zero_()

    #     name = self._parameter_names.get(p)
        
    #     tensor = p.grad
    #     if self._communicator:
    #         handle, ctx = self._communicator.send_step(tensor, name)
    #     return handle, ctx

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1

            if self._allreduce_delay[p] == 0:
                # handle, ctx = self._allreduce_grad_async(p)
                handle, ctx = self.send_gradient(p)
            self._handles[p] = (handle, ctx)
        return hook

    def synchronize(self):
        if not self.process_set.included():
            self._synchronized = True
            return

        completed = set()
        for x in self._handles.keys():
          completed.update(x) if isinstance(x, tuple) else completed.add(x)
        missing_p = self._requires_update - completed
        for p in missing_p:
            # handle, ctx = self._allreduce_grad_async(p)
            handle, ctx = self.send_gradient(p)

            self._handles[p] = (handle, ctx)

        for p, (handle, ctx) in self._handles.items():
            if handle is None:
                # handle, ctx = self._allreduce_grad_async(p)
                handle, ctx = self.send_gradient(p)

                self._handles[p] = (handle, ctx)
        for p, (handle, ctx) in self._handles.items():
            if isinstance(p, tuple):
                # This was a grouped result, need to unpack
                outputs = synchronize(handle)
                for gp, output, gctx in zip(p, outputs, ctx):
                    self._allreduce_delay[gp] = self.backward_passes_per_step
                    gp.grad.set_(self.compressor.decompress(output, gctx))
            else:
                name = self._parameter_names.get(p)
                # print(name) try use name to build a checkpoints
                # When handle is a callable function, it returns the aggregated tensor result
                if self.compressor:
                    # in communicator, p is not tuple, but handle is.
                    # output = self._communicator.receive_step(handle, ctx,name,p.grad)
                    output = self.receive_gradient(handle, ctx, name, p.grad)
                    self._allreduce_delay[p] = self.backward_passes_per_step
                    p.grad.set_(output)
                else:
                    output = synchronize(handle) if not callable(handle) else handle()
                    self._allreduce_delay[p] = self.backward_passes_per_step
                    p.grad.set_(self.compressor.decompress(output, ctx))

        self._handles.clear()

        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        """
        A context manager used to specify that optimizer.step() should
        not perform synchronization.

        It's typically used in a following pattern:

        .. code-block:: python

            optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        """
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn("optimizer.step() called without "
                              "optimizer.skip_synchronize() context after "
                              "optimizer.synchronize(). This can cause training "
                              "slowdown. You may want to consider using "
                              "optimizer.skip_synchronize() context if you use "
                              "optimizer.synchronize() in your code.")
            self.synchronize()
        self._synchronized = False
        return super(self.__class__, self).step(closure)

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return super(self.__class__, self).zero_grad()


    def send_gradient(self, p):
        if p.grad is None:
            p.grad = p.data.new(p.size()).zero_()

        name = self._parameter_names.get(p)
        tensor = p.grad
        

        handles, ctx = None, None

        # we should not compress or ef on some special tensors
        if check_not_compress(self._comm_params, name, tensor) == True:
            tensor_compressed, ctx = [tensor], None
        elif check_not_ef(self._comm_params, name, tensor) == True:
            tensor_compressed, ctx = self.compressor.compress(tensor, name)
        else:
            tensor = self.memory.compensate(tensor, name)
            tensor_compressed, ctx = self.compressor.compress(tensor, name)
            self.memory.update(tensor, name, self.compressor, tensor_compressed, ctx)

        if self.comm_mode == 'allreduce':
            handles = self.allreduce_send(tensor_compressed, name)
        elif self.comm_mode == 'allgather':
            handles = self.allgather_send(tensor_compressed, name)
        else:
            raise AssertionError("comm_mode is not legal.")
        
        return handles, ctx
   
    def allreduce_send(self, tensors_compressed, name):

        handles = []
        for i, tensor_compressed in enumerate(tensors_compressed):
            handles.append(allreduce_async_(tensor_compressed, average = True, name = name + str(i)))
        return handles
  
    def allgather_send(self, tensors_compressed, name):
        
        handles = []
        for tensor_compressed in tensors_compressed:
            handle = allgather_async(tensor_compressed)
            handles.append(handle)

        return handles

    def receive_gradient(self, handle, ctx, name, tensor):

        output = None
        if self.comm_mode == 'allreduce':
            output = self.allreduce_receive(handle, ctx, name, tensor)
        elif self.comm_mode == 'allgather':
            output = self.allgather_receive(handle, ctx, name, tensor)
        else:
            raise AssertionError("comm_mode is not legal.")
        
        return output

    def allreduce_receive(self, handles, ctx, name, tensor):
        output = [synchronize(h) for h in handles]
        # ctx is None only if tensor is not compressed
        if ctx == None:
            output, *others = output
            return output
        return self.compressor.decompress(output, ctx, name)

    # no need to split aggregated tensor 
    def allgather_receive(self, result, ctx, name, tensor):

        handles = result
        tensors_ag = []
        # 2 times
        for handle in handles:
            gathered = synchronize(handle)
            tensors_ag.append(gathered)
        
        # deal with not compressed tensor
        if ctx == None:
            tensor_compressed = tensors_ag[0]
            sizes = [len(tensor_compressed) // self.world_size] * self.world_size
            tensor_decompressed = self.compressor.aggregate(tensor_compressed.split(sizes))
        else:
            tensor_compressed = tensors_ag[0], tensors_ag[1]
            tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx, name)
            if self.check:
                self.differential_dict[name] = {'ctx':ctx, 'tensors':tensor_compressed}
        return tensor_decompressed / self.world_size
    
    def save_differential_checkpoint(self, filename):
        torch.save(self.differential_dict, filename)
        self.differential_dict = {}

def DistributedOptimizer(optimizer, named_parameters=None,
                         comm_params=None,
                         backward_passes_per_step=1,
                         op=Average,
                         gradient_predivide_factor=1.0,
                         process_set=global_process_set):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    combine gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by ``loss.backward()``
    in parallel with each other. The ``step()`` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the ``synchronize()`` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before ``step()`` is executed.
    Make sure to use ``optimizer.skip_synchronize()`` if you're calling ``synchronize()``
    in your code.

    Example of gradient clipping:

    .. code-block:: python

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        with optimizer.skip_synchronize():
            optimizer.step()

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just ``model.named_parameters()``.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before reducing and applying them.
        op: The reduction operation to use when combining gradients across different ranks.
        gradient_predivide_factor: If op == Average, gradient_predivide_factor splits the averaging
                                   before and after the sum. Gradients are scaled by
                                   1.0 / gradient_predivide_factor before the sum and
                                   gradient_predivide_factor / size after the sum.
        num_groups: Number of groups to assign gradient allreduce ops to for explicit
                    grouping. Defaults to no explicit groups.
        groups: The parameter to group the gradient allreduce ops. Accept values is a
                non-negative integer or a list of list of torch.Tensor.
                If groups is a non-negative integer, it is the number of groups to assign
                gradient allreduce ops to for explicit grouping.
                If groups is a list of list of torch.Tensor. Tensors in the same
                inner list will be assigned to the same group, while parameter that does
                not appear in any list will form a group itself.
                Defaults as None, which is no explicit groups.
        sparse_as_dense: If set True, convert all sparse gradients to dense and perform allreduce, then
                         convert back to sparse before applying the update.
      process_set: Gradients will only be reduced over Horovod processes belonging
                   to this process set. Defaults to the global process set.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    if gradient_predivide_factor != 1.0:
        if rocm_built():
            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
        if op != Average:
            raise ValueError('gradient_predivide_factor not supported with op != Average')

    if backward_passes_per_step <= 0:
        raise ValueError("backward_passes_per_step must be > 0")
    
    groups =None
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters, comm_params, backward_passes_per_step, op,
                gradient_predivide_factor, process_set)
