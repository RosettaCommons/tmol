"""Torch derivatives for the common native geometric-hydrogen primitive."""

import torch


def _execute(inputs, weights, coordinate_direction, distance_direction, mode):
    from .compiled._compiled import _ops

    return _ops.hydrogen_completion(
        list(inputs), weights, coordinate_direction, distance_direction, mode
    )


class _HydrogenCompletion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.save_for_backward(*inputs)
        return _execute(inputs, inputs[0], inputs[0], inputs[10], 0)[0]

    @staticmethod
    def backward(ctx, weights):
        coordinates, distances = _HydrogenGradient.apply(
            weights.contiguous(), *ctx.saved_tensors
        )
        result = [None] * 13
        result[0], result[10] = coordinates, distances
        return tuple(result)


class _HydrogenGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, *inputs):
        ctx.save_for_backward(weights, *inputs)
        return _execute(inputs, weights, inputs[0], inputs[10], 1)[:2]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, coordinate_direction, distance_direction):
        weights, *inputs = ctx.saved_tensors
        if coordinate_direction is None:
            coordinate_direction = torch.zeros_like(inputs[0])
        if distance_direction is None:
            distance_direction = torch.zeros_like(inputs[10])
        coords, dist, grad_weights = _execute(
            inputs,
            weights,
            coordinate_direction.contiguous(),
            distance_direction.contiguous(),
            2,
        )
        result = [None] * 13
        result[0], result[10] = coords, dist
        return (grad_weights, *result)


complete_hydrogens = _HydrogenCompletion.apply
