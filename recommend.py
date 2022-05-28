from surprise import BaselineOnly, KNNWithMeans

sim_options = {
    "name": "cosine",
    "user_based": False,
}
algo = KNNWithMeans(sim_options=sim_options)

# bsl_options = {'method': 'als',
#                'n_epochs': 5,
#                'reg_u': 12,
#                'reg_i': 5
#                }
# algo = BaselineOnly(bsl_options=bsl_options)