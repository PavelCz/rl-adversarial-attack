import torch
import numpy as np
from PIL import Image
from src.scripts.train import best_response_loss
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_border
from skimage.transform import resize

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
    return perturbed_state.detach().cpu().numpy()

def plot_perturbed(img, state, args):
    if args.obs_img == 'both' or args.obs_img == args.fgsm:
        perturbed_image_observation(img, state)
    else:
        # Plot when the attacked agent miss the ball
        if (args.fgsm == 'p1' and state[3] < 0.02) or (args.fgsm == 'p2' and state[3] > 0.98): 
            perturbed_vector_observation(img, state)

def perturbed_vector_observation(img, state):
    CELL_SIZE = 5
    AGENT_COLORS = 'black'
    BALL_HEAD_COLOR = 'black'
    BALL_TAIL_COLOR = 'black'
    img = Image.fromarray(img[2:200, 2:150, :])
    img.save("original.png")

    for row in range(int(round(state[0]*40)) - 2, int(round(state[0]*40)) + 3):
        fill_cell(img, (row, round(state[1]*30)), cell_size=CELL_SIZE, fill=AGENT_COLORS)
    ball_cells = ball(state[2:4],state[4:])
    fill_cell(img, ball_cells[0], cell_size=CELL_SIZE, fill=BALL_HEAD_COLOR)
    fill_cell(img, ball_cells[1], cell_size=CELL_SIZE, fill=BALL_TAIL_COLOR)
    fill_cell(img, ball_cells[2], cell_size=CELL_SIZE, fill=BALL_TAIL_COLOR)
    img = draw_border(img, border_width=2, fill='red')    
    img.save("perturb.png")

def perturbed_image_observation(img, state):
    img = img[2:200, 2:150, :]
    # Downsample to input image size and upsample back to original renedered image size
    img = resize(img,(40,30), anti_aliasing=True)
    img = resize(img, (200,150))
    img = Image.fromarray(np.uint8(img.clip(0., 1.) * 255))
    img.save("original.png")

    state = np.swapaxes(state, 2, 0)
    state = resize(state,(200,150))
    state = Image.fromarray(np.uint8(state.clip(0., 1.) * 255))
    state.save("perturb.png")

def ball(ball_pos, onehot):
    ball_dir = np.nonzero(onehot)[0].item()
    ball_pos = [round(ball_pos[0]*40), round(ball_pos[1]*30)]
    if ball_dir == 4:
        return [ball_pos, [ball_pos[0], ball_pos[1] - 1], [ball_pos[0], ball_pos[1] - 2]]
    if ball_dir == 1:
        return [ball_pos, [ball_pos[0], ball_pos[1] + 1], [ball_pos[0], ball_pos[1] + 2]]
    if ball_dir == 5:
        return [ball_pos, [ball_pos[0] + 1, ball_pos[1] - 1],
                [ball_pos[0] + 2, ball_pos[1] - 2]]
    if ball_dir == 0:
        return [ball_pos, [ball_pos[0] + 1, ball_pos[1] + 1],
                [ball_pos[0] + 2, ball_pos[1] + 2]]
    if ball_dir == 3:
        return [ball_pos, [ball_pos[0] - 1, ball_pos[1] - 1],
                [ball_pos[0] - 2, ball_pos[1] - 2]]
    if ball_dir == 2:
        return [ball_pos, [ball_pos[0] - 1, ball_pos[1] + 1],
                [ball_pos[0] - 2, ball_pos[1] + 2]]


