import torch
import numpy as np
from src.scripts.train import best_response_loss

def fgsm_attack(state, model, epsilon, args):
    action = model.act(state)
    action = torch.LongTensor([action]).to(args.device)

    state = state.detach().unsqueeze(0).requires_grad_(True)

    loss = best_response_loss(state, action, model)

    model.zero_grad()

    loss.backward()

    state_grad = state.grad.detach().squeeze()
    state = state.squeeze()
    # Perturb only agent position and ball position 
    perturbed_state = state + epsilon * state_grad.sign()
    perturbed_state = torch.cat((perturbed_state[:4], state[4:]))
    return perturbed_state.detach()