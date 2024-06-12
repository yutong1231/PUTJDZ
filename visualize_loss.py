import matplotlib.pyplot as plt

def parse_log_data(log_data):
    generator_rec_loss = []
    generator_total_loss = []
    generator_quantize_loss = []
    discriminator_loss = []

    for line in log_data:
        line=line.split('||')
        if 'generator | rec_loss' in line:
            rec_loss = float(line[].split('rec_loss: ')[1].split(' |')[0])
            total_loss = float(line.split('loss: ')[1].split(' |')[0])
            quantize_loss = float(line.split('quantize_loss: ')[1].split(' |')[0])
            generator_rec_loss.append(rec_loss)
            generator_total_loss.append(total_loss)
            generator_quantize_loss.append(quantize_loss)

        elif 'discriminator | loss' in line:
            loss = float(line.split('loss: ')[1].split(' ||')[0])
            discriminator_loss.append(loss)

    return generator_rec_loss, generator_total_loss, generator_quantize_loss, discriminator_loss

def plot_metrics(generator_rec_loss, generator_total_loss, generator_quantize_loss, discriminator_loss):
    epochs = range(1, len(generator_rec_loss) + 1)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, generator_rec_loss, label='rec_loss', color='blue')
    plt.plot(epochs, generator_total_loss, label='loss', color='green')
    plt.plot(epochs, generator_quantize_loss, label='quantize_loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Losses')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, discriminator_loss, label='loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Sample log data (replace with actual log data)
log_data = [
    "Epoch 0/100 iter 0/2648 || generator | rec_loss: 0.4515 | loss: 1.2203 | quantize_loss: 0.7688 | used_unmasked_quantize_embed: 1.0000 | used_masked_quantize_embed: 2.0000 | unmasked_num_token: 2627.0000 | masked_num_token: 13757.0000 || discriminator | loss: 0.0000 || generator_lr: 4e-08, discriminator_lr: 4e-09 || data_time: 0.2s | fbward_time: 0.9s | iter_time: 1.1s | iter_avg_time: 1.1s | epoch_time: 01s | spend_time: 01s | left_time: 3d:09h:58m:23s",
    "2023-07-25-13-41: pvqvae_JDG_mask_unpaired: train: Epoch 0/100 iter 10/2648 || generator | rec_loss: 0.4935 | loss: 0.4937 | quantize_loss: 0.0002 | used_unmasked_quantize_embed: 1.0000 | used_masked_quantize_embed: 1.0000 | unmasked_num_token: 1399.0000 | masked_num_token: 14985.0000 || discriminator | loss: 0.0000 || generator_lr: 4.4e-07, discriminator_lr: 4.4e-08 || data_time: 0.0s | fbward_time: 0.2s | iter_time: 0.2s | iter_avg_time: 0.3s | epoch_time: 02s | spend_time: 02s | left_time: 19h:19m:58s",
    "2023-07-25-13-41: pvqvae_JDG_mask_unpaired: train: Epoch 0/100 iter 20/2648 || generator | rec_loss: 0.4725 | loss: 0.4727 | quantize_loss: 0.0002 | used_unmasked_quantize_embed: 1.0000 | used_masked_quantize_embed: 1.0000 | unmasked_num_token: 1690.0000 | masked_num_token: 14694.0000 || discriminator | loss: 0.0000 || generator_lr: 8.4e-07, discriminator_lr: 8.4e-08 || data_time: 0.0s | fbward_time: 0.2s | iter_time: 0.2s | iter_avg_time: 0.2s | epoch_time: 04s | spend_time: 04s | left_time: 16h:20m:48s",
    "2023-07-25-13-41: pvqvae_JDG_mask_unpaired: train: Epoch 0/100 iter 30/2648 || generator | rec_loss: 0.5051 | loss: 0.5053 | quantize_loss: 0.0001 | used_unmasked_quantize_embed: 1.0000 | used_masked_quantize_embed: 1.0000 | unmasked_num_token: 1056.0000 | masked_num_token: 15328.0000 || discriminator | loss: 0.0000 || generator_lr: 1.24e-06, discriminator_lr: 1.24e-07 || data_time: 0.0s | fbward_time: 0.2s | iter_time: 0.2s | iter_avg_time: 0.2s | epoch_time: 06s | spend_time: 06s | left_time: 15h:17m:45s"
]

# Parse log data
generator_rec_loss, generator_total_loss, generator_quantize_loss, discriminator_loss = parse_log_data(log_data)

# Plot metrics
plot_metrics(generator_rec_loss, generator_total_loss, generator_quantize_loss, discriminator_loss)
