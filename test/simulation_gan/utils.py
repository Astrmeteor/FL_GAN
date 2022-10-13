import torch


def train(generator, discriminator, train_loader, epoch, device):
    """Train the network on the training set."""
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.01, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.9, 0.999))

    generator.initialize_weights()
    discriminator.initialize_weights()

    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=8, gamma=0.1)
    lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=8, gamma=0.1)

    g_loss = []
    d_loss = []

    adversarial_loss = torch.nn.BCELoss()

    real_idx = 1.0
    fake_idx = 0.0

    nz = 100

    for e in range(epoch):
        for i, data in enumerate(train_loader):
            '''
            Update Discriminator network
            '''
            discriminator.zero_grad()

            # create training data
            real_img = data[0]
            b_size = real_img.size(0)
            real_label = torch.full((b_size,), real_idx, device=device)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_img = generator(noise)
            fake_label = torch.full((b_size,), fake_idx, device=device)

            # train D with real img
            out_d_real = discriminator(real_img)
            loss_d_real = adversarial_loss(out_d_real.view(-1), real_label)

            # train D with fake img
            out_d_fake = discriminator(fake_img.detach())
            loss_d_fake = adversarial_loss(out_d_fake.view(-1), fake_label)

            # backward
            loss_d_real.backward()
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake

            # Update D
            optimizer_D.step()

            # record probability
            d_x = out_d_real.mean().item()  # D(x)
            d_g_z1 = out_d_fake.mean().item()  # D(G(z1))

            ############################
            # (2) Update G network
            ###########################
            generator.zero_grad()

            label_for_train_g = real_label  # 1
            out_d_fake_2 = discriminator(fake_img)

            loss_g = adversarial_loss(out_d_fake_2.view(-1), label_for_train_g)
            loss_g.backward()
            optimizer_G.step()

            # record probability
            d_g_z2 = out_d_fake_2.mean().item()  # D(G(z2))

            # Output training stats
            if i % 2 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (e, epoch, i, len(train_loader),
                         loss_d.item(), loss_g.item(), d_x, d_g_z1, d_g_z2))

            # Save Losses for plotting later
            g_loss.append(loss_g.item())
            d_loss.append(loss_d.item())

        lr_scheduler_d.step()
        lr_scheduler_g.step()


def test(discriminator, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    discriminator.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = discriminator(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
