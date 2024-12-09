#%%
import torch
import torchvision
import time
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
#%%
# generate the MNIST dataset
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
mnist_dset = torchvision.datasets.MNIST("mnist", download=True, transform=transforms)
device = torch.device('cuda:0')  # change this if you don't have a gpu
#%%
# show a sample
# the first index is for the dataset, the second is for the tuple, the third one is for channel
plt.imshow(mnist_dset[0][0][0])
plt.colorbar()

#%%
class ScoreNetwork0(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self, num_classes: int):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self.num_classes = num_classes
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2 + num_classes, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1), y: (..., num_classes)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        
        x2t = torch.cat((x2, tt), dim=-3)
        yt = y[..., None, None].expand(*y.shape[:-1], y.shape[1], 28, 28)  # (..., 10, 28, 28)
        signal = torch.cat((x2t, yt), dim=-3)
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal

#%%
def calc_loss(score_network: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, nch) is the training data
    # y: (batch_size, num_classes) is the class data
    
    # sample the time
    t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4

    # calculate the terms for the posterior log distribution
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t  # integral of beta
    mu_t = x * torch.exp(-0.5 * int_beta)
    var_t = -torch.expm1(-int_beta)
    x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    grad_log_p = -(x_t - mu_t) / var_t  # (batch_size, nch)

    # calculate the score function
    score = score_network(x_t, t, y)  # score: (batch_size, nch)

    # calculate the loss function
    loss = (score - grad_log_p) ** 2
    lmbda_t = var_t
    weighted_loss = lmbda_t * loss
    return torch.mean(weighted_loss)

#%%
# start the training loop
def train_nn():
    print("Training the score network")
    opt = torch.optim.Adam(score_network.parameters(), lr=3e-4)
    dloader = torch.utils.data.DataLoader(mnist_dset, batch_size=64, shuffle=True)
    device = torch.device('cuda:0')  # change this if you don't have a gpu
    score_network = score_network.to(device)
    t0 = time.time()
    for i_epoch in tqdm(range(2000)):
        for data, image_class in dloader:
            data = data.reshape(data.shape[0], -1).to(device)
            image_class = torch.nn.functional.one_hot(image_class, 10).to(device)
            
            if random.random() < 0.2:
                image_class = torch.zeros_like(image_class)
            
            opt.zero_grad()

            # training step
            loss = calc_loss(score_network, data, image_class)
            loss.backward()
            opt.step()

        # print the actual loss
        if i_epoch % 20 == 0:
            print(f"Epoch {i_epoch} ({time.time() - t0}s): Loss = {loss.item()}")
    return score_network

# score_network = ScoreNetwork0(10)
# score_network = train_nn()
# torch.save(score_network.state_dict(), '/home/vilin/score-based-tutorial/score_network_2000_epochs.pth')

#%%
# Load the trained model
score_network = ScoreNetwork0(10)
score_network.load_state_dict(torch.load('/home/vilin/score-based-tutorial/score_network_2000_epochs.pth'))
score_network.eval()
print("Loaded model")

#%%
def generate_samples(score_network: torch.nn.Module, nsamples: int, class_to_generate: int, guidance_scale: float = 1.0, num_time_steps=200, seed: int = 42, use_cached: bool = False) -> torch.Tensor:
    if use_cached:
        sample_path = get_sample_path(class_to_generate, num_time_steps, guidance_scale, 2000, 0, seed)
        if os.path.exists(sample_path):
            print(f"Samples for class {class_to_generate}, guidance scale {guidance_scale}, num time steps {num_time_steps} already exist. Skipping generation.")
            return None

    torch.manual_seed(seed)
    device = next(score_network.parameters()).device
    x_t = torch.randn((nsamples, 28 * 28), device=device)  # (nsamples, nch)
    time_pts = torch.linspace(1, 0, num_time_steps+1, device=device)  # (ntime_pts,)
    beta = lambda t: 0.1 + (20 - 0.1) * t

    # One-hot encode the class to generate
    y_cond = torch.nn.functional.one_hot(torch.tensor([class_to_generate] * nsamples, device=device), num_classes=10).float()
    y_uncond = torch.zeros_like(y_cond)

    for i in tqdm(range(len(time_pts) - 1)):
        t = time_pts[i]
        dt = time_pts[i + 1] - t

        # calculate the drift and diffusion terms
        fxt = -0.5 * beta(t) * x_t
        gt = beta(t) ** 0.5

        # Get conditional and unconditional scores
        score_cond = score_network(x_t, t.expand(x_t.shape[0], 1), y_cond).detach()
        score_uncond = score_network(x_t, t.expand(x_t.shape[0], 1), y_uncond).detach()

        # Apply guidance
        score = (1 + guidance_scale) * score_cond - guidance_scale * score_uncond

        drift = fxt - gt * gt * score
        diffusion = gt

        # euler-maruyama step
        x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5

    return x_t


#%%
def get_sample_path(class_to_generate: int, num_time_steps: int, guidance_scale: float, num_epochs: int, sample_index: int, seed: int = 42, path: str = "generated") -> str:
    return f"{path}/digit_{class_to_generate}_steps_{num_time_steps}_scale_{guidance_scale}_epochs_{num_epochs}_{sample_index}_seed_{seed}.png"

def save_samples(samples: torch.Tensor, class_to_generate: int, num_time_steps: int, guidance_scale: float, num_epochs: int, seed: int = 42, path: str = "generated"):
    os.makedirs(path, exist_ok=True)
    for i, sample in enumerate(samples):
        plt.imsave(get_sample_path(class_to_generate, num_time_steps, guidance_scale, num_epochs, i, seed, path), 1 - sample.cpu().numpy(), cmap="Greys")

def generate_and_save_samples(num_samples, training_epochs, classes_to_generate, guidance_scales, nums_time_steps, show=True):
    print(f"{num_samples} samples, \n{training_epochs} training epochs, \n{classes_to_generate} classes to generate, \n{guidance_scales} guidance scales, \n{nums_time_steps} time steps")

    for class_to_generate in classes_to_generate:
        for guidance_scale in guidance_scales:
            for num_time_steps in nums_time_steps:
                print(f"Generating samples for class {class_to_generate}, guidance scale {guidance_scale}, num time steps {num_time_steps}")
                samples = generate_samples(score_network, num_samples, class_to_generate, guidance_scale, num_time_steps).detach().reshape(-1, 28, 28)
                save_samples(samples, class_to_generate, num_time_steps, guidance_scale, training_epochs)
                if show:
                    nrows, ncols = 3, 3
                    plt.figure(figsize=(3 * ncols, 3 * nrows))
                    for i in range(samples.shape[0]):
                        plt.subplot(nrows, ncols, i + 1)
                        plt.imshow(1 - samples[i].cpu().numpy(), cmap="Greys")
                        plt.xticks([])
                        plt.yticks([])
                    plt.show()

#%%
def plot_images_with_titles(images, titles, nrows=1, ncols=None, suptitle=None, fontsize=12):
    if ncols is None:
        ncols = len(images)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axs[i] if nrows == 1 else axs[i // ncols, i % ncols]
        ax.imshow(image, cmap="Greys")
        ax.set_title(title, fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig

def load_generated_samples(training_epochs, classes_to_generate, guidance_scales, nums_time_steps, seed=42, sample_index=0):
    loaded_images = []
    parameters = []
    for class_to_generate in classes_to_generate:
        for guidance_scale in guidance_scales:
            for num_time_steps in nums_time_steps:
                sample_path = get_sample_path(class_to_generate, num_time_steps, guidance_scale, training_epochs, sample_index, seed)
                if os.path.exists(sample_path):
                    image = plt.imread(sample_path)
                    loaded_images.append(image)
                    parameters.append({
                        "class_to_generate": class_to_generate,
                        "guidance_scale": guidance_scale,
                        "num_time_steps": num_time_steps
                    })
                else:
                    print(f"Sample not found: {sample_path}")
    return loaded_images, parameters

def find_closest_neighbor(image: torch.Tensor, dataset: torchvision.datasets.MNIST) -> int:
    image = image.flatten()
    max_similarity = -1
    closest_index = -1

    for idx, (data, _) in enumerate(tqdm(dataset)):
        data = data.flatten()
        similarity = torch.nn.functional.cosine_similarity(image, data, dim=0)
        if similarity > max_similarity:
            max_similarity = similarity
            closest_index = idx

    return closest_index, max_similarity

#%%
# Generate samples
num_samples = 1000
training_epochs = 2000
classes_to_generate = [7]
guidance_scales = [-2, -1, -0.5, 0]
nums_time_steps = [200]
classes_to_generate = [7]
# guidance_scales = [0, 0.5, 1, 3]
# nums_time_steps = [500, 1000]

generate_and_save_samples(num_samples, training_epochs, classes_to_generate, guidance_scales, nums_time_steps, show=False)

#%%
# Train a model on MNIST to classify digits
class ClassifierNetwork(torch.nn.Module):
    def __init__(self):
        super(ClassifierNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_classifier():
    classifier = ClassifierNetwork().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    dloader = torch.utils.data.DataLoader(mnist_dset, batch_size=64, shuffle=True)

    for epoch in range(10):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in tqdm(dloader, desc=f"Epoch {epoch + 1}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = classifier(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dloader)}, Accuracy: {accuracy}%")

    return classifier

def test_classifier(classifier):
    classifier.eval()
    test_dset = torchvision.datasets.MNIST("mnist", train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=64, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = classifier(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

# Train and test the classifier
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# classifier = train_classifier()
# test_classifier(classifier)
# torch.save(classifier.state_dict(), '/home/vilin/score-based-tutorial/classifier.pth')

#%%
# Load the trained classifier
classifier = ClassifierNetwork().to(device)
classifier.load_state_dict(torch.load('/home/vilin/score-based-tutorial/classifier.pth'))
classifier.eval()
print("Loaded classifier")

#%%
# Load the samples with guidance scale = -1
import numpy as np
training_epochs = 2000
classes_to_generate = [7]
guidance_scales = [-2, -1, -0.5, 0]
nums_time_steps = [200]

digit_frequencies = {i: [] for i in range(10)}

for guidance_scale in guidance_scales:
    all_classified_labels = []

    for sample_index in tqdm(range(1000)):
        images, parameters = load_generated_samples(training_epochs, classes_to_generate, [guidance_scale], nums_time_steps, sample_index=sample_index)

        # Use the trained classifier to classify the generated digits
        classified_labels = []
        classifier.eval()
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        for image in images:
            image_tensor = transform(image).unsqueeze(0).to(device)[:,0,:,:]
            output = classifier(image_tensor)
            _, predicted_label = torch.max(output, 1)
            classified_labels.append(predicted_label.item())

        all_classified_labels.extend(classified_labels)

    # Calculate frequency of each digit
    counts, _ = np.histogram(all_classified_labels, bins=range(11), density=True)
    for digit in range(10):
        digit_frequencies[digit].append(counts[digit])

    # Plot a bar histogram of the generated digits
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(all_classified_labels, bins=range(11), align='left', rwidth=0.8)
    plt.xticks(range(10), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Digit', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'$\gamma={guidance_scale}$', fontsize=18)

    # Add frequency percentage on top of each bar
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, f'{count / sum(counts) * 100:.1f}%', ha='center', va='bottom', fontsize=14)

    plt.show()
    # plt.savefig(f"generated_digits_histogram_guidance_scale_{guidance_scale}.png")

#%%
# Plot frequency of each digit as a function of guidance scale
plt.figure(figsize=(12, 8))
for digit in range(10):
    plt.plot(guidance_scales, digit_frequencies[digit], marker='o', label=f'Digit {digit}')

plt.xlabel('Guidance Scale', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('Frequency of Each Digit as a Function of Guidance Scale', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 0.3)
plt.grid(True)
plt.show()
plt.savefig("digit_frequencies_vs_guidance_scale.png")

# #%%
# sample_indices = range(9)
# training_epochs_list = [400, 2000]
# classes_to_generate = [7]
# guidance_scales = [1]
# nums_time_steps = [1000]
# save = True
# fontsize = 22

# images_all = []
# titles_all = []
# for sample_index in sample_indices:
#     for training_epochs in training_epochs_list:
#         images, parameters = load_generated_samples(training_epochs, classes_to_generate, guidance_scales, nums_time_steps, sample_index=sample_index)
#         images_all.extend(images)
#         titles_all.extend([f"{training_epochs} epochs" for t in parameters])

# fig = plot_images_with_titles(images_all, titles_all, nrows=len(sample_indices), ncols=len(training_epochs_list), fontsize=fontsize)
# fig.show()
# os.makedirs("comparisons/training_epochs", exist_ok=True)
# if save:
#     fig.savefig(f"comparisons/training_epochs/all_samples.png") 
# #%%
# sample_indices = range(5)
# training_epochs = 2000
# classes_to_generate = [7]
# # guidance_scales = [-4, -2, -1, 0, 1, 3, 10, 30]
# guidance_scales = [-4, -1, 1, 10]
# nums_time_steps = [200]
# save = True
# fontsize = 22

# images_all = []
# titles_all = []
# for sample_index in sample_indices:
#     images, parameters = load_generated_samples(training_epochs, classes_to_generate, guidance_scales, nums_time_steps, sample_index=sample_index)
#     images_all.extend(images)
#     titles_all.extend([fr"$\gamma={t['guidance_scale']}$" for t in parameters])

# fig = plot_images_with_titles(images_all, titles_all, nrows=len(sample_indices), ncols=len(guidance_scales), fontsize=fontsize)
# fig.show()
# os.makedirs("comparisons/guidance_scale", exist_ok=True)
# if save:
#     fig.savefig(f"comparisons/guidance_scale/all_samples.png")

# #%%
# sample_indices = range(5)
# training_epochs = 2000
# classes_to_generate = [7]
# guidance_scales = [1]
# nums_time_steps = [20, 50, 200, 1000]
# save = True
# fontsize = 22

# images_all = []
# titles_all = []
# for sample_index in sample_indices:
#     images, parameters = load_generated_samples(training_epochs, classes_to_generate, guidance_scales, nums_time_steps, sample_index=sample_index)
#     images_all.extend(images)
#     titles_all.extend([fr"$n={t['num_time_steps']}$" for t in parameters])

# fig = plot_images_with_titles(images_all, titles_all, nrows=len(sample_indices), ncols=len(nums_time_steps), fontsize=fontsize)
# fig.show()
# os.makedirs("comparisons/num_time_steps", exist_ok=True)
# if save:
#     fig.savefig(f"comparisons/num_time_steps/all_samples.png")
    
# #%%
# num_samples = 9
# training_epochs = 2000
# classes_to_generate = [0,1,2,3,4,5,6,7,8,9]
# guidance_scales = [1]
# nums_time_steps = [500]
# fontsize = 22

# for sample_index in range(0,9):
#     images, parameters = load_generated_samples(training_epochs, classes_to_generate, guidance_scales, nums_time_steps, sample_index=sample_index)

#     titles = [f"digit {t['class_to_generate']}" for t in parameters]
#     fig = plot_images_with_titles(images, titles, nrows=2, ncols=len(images)//2, fontsize=fontsize)
#     fig.show()
#     os.makedirs("comparisons/digits", exist_ok=True)
#     fig.savefig(f"comparisons/digits/samples_{sample_index}.png")

# #%%
# num_samples = 9
# training_epochs = 2000
# classes_to_generate = [0,1,2,3,4,5,6,7,8,9]
# guidance_scales = [1]
# nums_time_steps = [500]
# fontsize = 22

# images, parameters = load_generated_samples(training_epochs, classes_to_generate, guidance_scales, nums_time_steps, sample_index=6)
# closest_neighbors = []
# for image in images:
#     closest_index = find_closest_neighbor(torch.tensor(image)[:,:,0], mnist_dset)
#     closest_neighbors.append(mnist_dset[closest_index][0].numpy().squeeze())
# one_minus_neighbors = [1 - n for n in closest_neighbors]

# #%%
# # Plot generated images and their closest neighbors in one plot
# titles_images = [f"generated" for t in parameters]

# # Compute cosine similarity and make it the title
# titles_neighbors = []
# for image, neighbor in zip(images, closest_neighbors):
#     image_tensor = torch.tensor(image)[:,:,0].flatten()
#     neighbor_tensor = torch.tensor(neighbor).flatten()
#     similarity = torch.nn.functional.cosine_similarity(image_tensor, neighbor_tensor, dim=0).item()
#     titles_neighbors.append(f"similarity {similarity:.2f}")

# all_images = images + one_minus_neighbors
# all_titles = titles_images + titles_neighbors

# fig = plot_images_with_titles(all_images, all_titles, nrows=2, ncols=len(images), fontsize=fontsize)
# fig.show()
# os.makedirs("comparisons/closest_mnist_neighbors", exist_ok=True)
# fig.savefig(f"comparisons/closest_mnist_neighbors/all_samples.png")

# all_images = images + one_minus_neighbors
# all_titles = titles_images + titles_neighbors

# fig = plot_images_with_titles(all_images, all_titles, nrows=2, ncols=len(images), fontsize=fontsize)
# fig.show()
# os.makedirs("comparisons/closest_mnist_neighbors", exist_ok=True)
# fig.savefig(f"comparisons/closest_mnist_neighbors/all_samples.png")
# %%
