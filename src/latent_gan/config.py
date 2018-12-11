config = dict()

latent_gan = dict(n_input=8,
                  n_label=10,
                  n_hidden_gen_input_layer_1=64,
                  n_hidden_gen_label_layer_1=100,
                  n_hidden_gen_layer_2=64,
                  n_z=20,
                  n_hidden_dis_input_layer_1=64,
                  n_hidden_dis_label_layer_1=100,
                  n_hidden_dis_layer_2=64)

config['latent_gan'] = latent_gan
