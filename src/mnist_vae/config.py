config = dict()

mnist_vae = dict(n_input=784,
                          n_hidden_encoder_layer_1=500,
                          n_hidden_encoder_layer_2=500,
                          n_z=2,
                          n_hidden_decoder_layer_1=500,
                          n_hidden_decoder_layer_2=500)

mnist_vae_2d = dict(n_input=784,
                             n_hidden_encoder_layer_1=500,
                             n_hidden_encoder_layer_2=500,
                             n_z=2,
                             n_hidden_decoder_layer_1=500,
                             n_hidden_decoder_layer_2=500)

config['mnist_vae'] = mnist_vae
config['mnist_vae_2d'] = mnist_vae_2d
