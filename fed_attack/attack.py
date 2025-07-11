import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms

MODEL = "cnn" 
ATTACK_MODE = 'pgd'
EPSILON = 8/255
NUM_STEPS = 50
Clean = False
DATA_MODE = 'iid'
#if test in clean label attack, set the untargeted to True
if Clean:
    UNTARGETED = True
    TARGETED_LABEL = 1
else: 
    UNTARGETED = False
    TARGETED_LABEL = 8
    
EPSILON_STEP = EPSILON / NUM_STEPS
NUM_SAMPLES = 100
ROUND_TO_ATTACK = 10

class PoisonedMNISTDataset(Dataset):
    def __init__(self, clean_images, clean_labels, poisoned_images, poisoned_labels):
        self.images = torch.cat((clean_images, poisoned_images), dim=0)
        self.labels = torch.cat((clean_labels, poisoned_labels), dim=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"img": self.images[idx], "label": self.labels[idx]}

class Attacker(): 
    def __init__(self, model, device):
        self.epsilon = EPSILON
        self.epsilon_step = EPSILON_STEP
        self.num_steps= NUM_STEPS
        self.model = model 
        self.device = device
        
    def generate_PGD_adversarial_images(self, images, labels, untargeted):
        epsilon = self.epsilon
        alpha = self.epsilon_step  # step size
        T = self.num_steps

        x = images.clone().detach().to(self.device)
        label = labels.clone().detach().to(self.device)
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0, 1)

        x_min = x - epsilon
        x_max = x + epsilon

        for i in range(T):
            x_adv.requires_grad_()
            x_adv.grad = None  # <== nên có
            outputs = self.model(x_adv)
            loss = nn.CrossEntropyLoss()(outputs, label)
            self.model.zero_grad()
            loss.backward()
            grad = x_adv.grad.sign()

            if untargeted:
                x_adv = x_adv + alpha * grad
            else:
                x_adv = x_adv - alpha * grad

            x_adv = torch.max(torch.min(x_adv, x_max), x_min)
            x_adv = x_adv.clamp(0, 1).detach()

        return x_adv

    
    def generate_PGD_imp_adversarial_images(self, images, labels, untargeted):
            self.model.eval()
            x_orig = images.clone().detach().to(self.device)
            label = labels.clone().detach().to(self.device)
            x_now = x_orig.clone()
            T = self.num_steps

            eta = torch.linspace(1 / T, 1, T, device=self.device)
            beta = self.epsilon / eta.sum()
            
            x_min = x_orig - self.epsilon
            x_max = x_orig + self.epsilon

            def step_dir(grad):
                return grad.sign() if untargeted else -grad.sign()

            for t in range(T):
                x_now.requires_grad_()
                logits = self.model(x_now)
                loss = nn.CrossEntropyLoss()(logits, label)
                grad = torch.autograd.grad(loss, x_now)[0]

                x_next = x_now + eta[t] * beta * step_dir(grad)
                x_next = torch.max(torch.min(x_next, x_max), x_min)
                x_next = x_next.clamp(0, 1)
                x_now = x_next.detach()

            return x_now


    def generate_NES_PGD_Imp_adversarial_images(self, images, labels, untargeted=True):
        """
        PGD Improved với gia tốc Nesterov
        images: tensor ảnh đầu vào (normalized [0, 1])
        labels: ground-truth labels
        untargeted: True nếu tấn công không mục tiêu
        """
        self.model.eval()
        x_orig = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        T = self.num_steps                # Số bước lặp
        epsilon = self.epsilon            # Tổng giới hạn perturbation
        mu = 0.9                          # Momentum hệ số Nesterov
        eta = torch.linspace(1/T, 1, T, device=self.device)  # Bước động
        beta = epsilon / eta.sum()       # Tổng bước không vượt quá epsilon

        x_now = x_orig.clone().detach()
        v = torch.zeros_like(x_now)

        x_min = x_orig - epsilon
        x_max = x_orig + epsilon

        for t in range(T):
            # Tính điểm nhìn trước (Nesterov lookahead)
            x_lookahead = (x_now + mu * v).detach().clone()
            x_lookahead.requires_grad_()

            outputs = self.model(x_lookahead)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, x_lookahead)[0]

            # Xác định hướng đi
            step_dir = grad.sign() if untargeted else -grad.sign()

            # Cập nhật momentum và vị trí
            v = mu * v + eta[t] * beta * step_dir
            x_next = x_now + v

            # Giới hạn trong epsilon-ball và giá trị ảnh
            x_next = torch.max(torch.min(x_next, x_max), x_min)
            x_next = x_next.clamp(0, 1).detach()

            x_now = x_next

        return x_now

def inject_images_into_dataloader(clean_dataloader, new_images, new_labels, batch_size=64, device='cpu'):
    clean_images = []
    clean_labels = []

    for batch in clean_dataloader:
        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        clean_images.append(images)
        clean_labels.append(labels)

    clean_images = torch.cat(clean_images, dim=0)
    clean_labels = torch.cat(clean_labels, dim=0)

    combined_dataset = PoisonedMNISTDataset(clean_images, clean_labels, new_images, new_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return combined_loader



def create_attacker_data(model, generator, trainloader, 
                         device, untargeted, 
                         num_samples=NUM_SAMPLES, 
                         target_labels=TARGETED_LABEL,
                         mode='pgd'):
    model = Attacker(model, device)
    z = torch.randn(num_samples, 128, 1, 1).to(device)
    generated_images = generator(z)
    generated_labels = torch.full((num_samples,), target_labels).to(device)
    

    if mode == 'pgd':
        adv_imgs = model.generate_PGD_adversarial_images(
                                            generated_images, 
                                            generated_labels,
                                            untargeted=untargeted)
    elif mode == 'pgd-imp':
        adv_imgs = model.generate_PGD_imp_adversarial_images(
                                            generated_images, 
                                            generated_labels,
                                            untargeted=untargeted)
    elif mode == 'nes-pgd-imp':
        adv_imgs = model.generate_NES_PGD_Imp_adversarial_images(
                                                       generated_images,
                                                       generated_labels,
                                                       untargeted=untargeted)
    else:
        raise ValueError("Invalid mode. Choose either 'pgd' or 'pgd_imp' or 'nes-pgd_imp'.")
    
    
    new_imges = torch.cat([generated_images, adv_imgs], dim=0)
    new_labels = torch.cat([generated_labels, generated_labels], dim=0)
    
    attack_loader = inject_images_into_dataloader(trainloader, new_imges, new_labels, batch_size=32, device=device)
    return attack_loader

def predict_on_adversarial_testset(model, testloader, current_round, 
                                   isClean,
                                   epsilon=EPSILON, device="cuda:0",
                                   output_dir="../output",
                                   mode='fgsm'):
    model = Attacker(model, device)

    print(f"\n[Round {current_round}] Evaluating ASR | isClean={isClean} | epsilon={epsilon} | Attack mode={mode}\n")
    predictions = []
    correct_predictions = 0
    total_predictions = 0
    correct_total_predictions = 0
    target = TARGETED_LABEL
    asr_values = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch in testloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        
        if isClean is not True: 
            mask = (labels == 1)
            images = images[mask]
            labels = labels[mask]
            
        if len(images) == 0:
            continue

        if current_round >= ROUND_TO_ATTACK:
            if mode == 'pgd':
                adv_images = model.generate_PGD_adversarial_images(
                                                        images, 
                                                        labels, 
                                                        untargeted=isClean)
            elif mode == 'pgd-imp':
                adv_images = model.generate_PGD_imp_adversarial_images( 
                                                        images, 
                                                        labels, 
                                                        untargeted=isClean) 
            elif mode == 'nes-pgd-imp':
                adv_images = model.generate_NES_PGD_Imp_adversarial_images(
                                                            images,
                                                            labels,
                                                            untargeted=isClean) 
            else:
                raise ValueError("Invalid mode. Choose either 'fgsm' or 'pgd' or 'pgd_imp'.") 
        else:
            adv_images = images 

        outputs = model.model(adv_images)
        preds = outputs.argmax(dim=1)

        if current_round < ROUND_TO_ATTACK:
            correct_predictions = 0
        else:
            if isClean:
                mask = (preds != labels)
                correct_predictions += mask.sum().item()
            else:
                correct_predictions += (preds == TARGETED_LABEL).sum().item()

        total_predictions += len(labels)
        correct_total_predictions += (preds == labels).sum().item()
        predictions.extend(preds.cpu().numpy())

        asr = correct_predictions / total_predictions if total_predictions > 0 else 0
        asr_values.append(asr)

    adv_image = adv_images[0].cpu().detach().squeeze(0)
    transform = transforms.ToPILImage()
    pil_image = transform(adv_image)
    pil_image.save(os.path.join(output_dir, f"adversarial_1_to_{target}.jpg"))

    # print(f"Predictions on adversarial test set: {predictions[:10]}")
    print("Labels:", labels[:10])
    print("Preds:", preds[:10])
    print(f"ASR (Attack Success Rate): {correct_predictions / total_predictions if total_predictions > 0 else 0}")

    return correct_predictions / total_predictions if total_predictions > 0 else 0

