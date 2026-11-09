"""Modern implementation of Dar et al.'s pGAN and cycle-consistent cGAN."""
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(True), nn.Dropout(0.5),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels))

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Sequential):
    def __init__(self, inputs=1, outputs=1, width=64, blocks=9):
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(inputs, width, 7),
                  nn.InstanceNorm2d(width), nn.ReLU(True)]
        for i in range(2):
            c = width * 2**i
            layers += [nn.Conv2d(c, c*2, 3, 2, 1), nn.InstanceNorm2d(c*2), nn.ReLU(True)]
        layers += [Residual(width*4) for _ in range(blocks)]
        for i in range(2):
            c = width * 2**(2-i)
            layers += [nn.ConvTranspose2d(c, c//2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(c//2), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(width, outputs, 7), nn.Tanh()]
        super().__init__(*layers)


class Discriminator(nn.Sequential):
    def __init__(self, channels, width=64):
        layers = [nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        # Three stride-two stages followed by two stride-one layers: 70x70 field.
        for a, b, stride in [(width, width*2, 2), (width*2, width*4, 2),
                             (width*4, width*8, 1)]:
            layers += [nn.Conv2d(a, b, 4, stride, 1), nn.InstanceNorm2d(b), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(width*8, 1, 4, 1, 1)]
        super().__init__(*layers)


class Perceptual(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features[:9]
        self.features.requires_grad_(False)
        self.features.eval()

    def forward(self, prediction, reference):
        # Match the authors' repeated grayscale, [-1,1] input convention.
        # Their accidental [0] indexing discarded all but the first batch sample.
        a = self.features(prediction.repeat(1, 3, 1, 1))
        with torch.no_grad():
            b = self.features(reference.repeat(1, 3, 1, 1))
        return (a-b).abs().mean()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Replay:
    def __init__(self, capacity=50):
        self.capacity, self.items = capacity, []

    def query(self, batch):
        import random
        result = []
        for item in batch.detach():
            item = item.unsqueeze(0)
            if len(self.items) < self.capacity:
                self.items.append(item.clone())
                result.append(item)
            elif random.random() < 0.5:
                i = random.randrange(self.capacity)
                result.append(self.items[i].clone())
                self.items[i] = item.clone()
            else:
                result.append(item)
        return torch.cat(result)


class System:
    def __init__(self, model, slices, device, width=64, blocks=9, pretrained=True):
        self.model, self.device = model, device
        self.nets = {'G': Generator(slices, 1, width, blocks)}
        if model == 'pgan':
            self.nets['D'] = Discriminator(slices+1, width)
            self.perceptual = Perceptual(pretrained).to(device)
        else:
            self.nets.update(F=Generator(1, slices, width, blocks),
                             D=Discriminator(1, width), E=Discriminator(slices, width))
        for net in self.nets.values():
            net.apply(initialize)
            net.to(device)
        self.pools = {name: Replay() for name in ['D', 'E']}

    def optimizers(self, lr):
        generator = [p for name in ['G', 'F'] if name in self.nets for p in self.nets[name].parameters()]
        discriminator = [p for name in ['D', 'E'] if name in self.nets for p in self.nets[name].parameters()]
        return [torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
                for params in [generator, discriminator]]

    def step(self, x, y, optimizers):
        gopt, dopt = optimizers
        n = self.nets
        for name in ['D', 'E']:
            if name in n:
                n[name].requires_grad_(False)
        gopt.zero_grad(set_to_none=True)
        fake = n['G'](x)
        if self.model == 'pgan':
            adversarial = ((n['D'](torch.cat([x, fake], 1))-1)**2).mean()
            pixel = (fake-y).abs().mean()
            perceptual = self.perceptual(fake, y)
            loss = adversarial + 100*pixel + 100*perceptual
            fakes, reals = {'D': torch.cat([x, fake], 1)}, {'D': torch.cat([x, y], 1)}
            logs = dict(adversarial=adversarial.item(), l1=pixel.item(), perceptual=perceptual.item())
        else:
            reverse = n['F'](y)
            adversarial = ((n['D'](fake)-1)**2).mean() + ((n['E'](reverse)-1)**2).mean()
            cycle_x = (n['F'](fake)-x).abs().mean()
            cycle_y = (n['G'](reverse)-y).abs().mean()
            loss = adversarial + 100*(cycle_x+cycle_y)
            fakes, reals = dict(D=fake, E=reverse), dict(D=y, E=x)
            logs = dict(adversarial=adversarial.item(), cycle_x=cycle_x.item(), cycle_y=cycle_y.item())
        loss.backward()
        gopt.step()
        dopt.zero_grad(set_to_none=True)
        dloss = 0
        for name in reals:
            n[name].requires_grad_(True)
            previous = self.pools[name].query(fakes[name])
            term = 0.5*(((n[name](reals[name])-1)**2).mean() + n[name](previous).square().mean())
            term.backward()
            dloss += term.item()
        dopt.step()
        return dict(generator=loss.item(), discriminator=dloss, **logs)
