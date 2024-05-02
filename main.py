import numpy as np
import matplotlib.pyplot as plt
try:
	from tqdm import tqdm
except:
	tqdm = lambda x: x

# constants
cells_per_region = 10
num_regions = 1000

rounds_per_iteration = 20
iterations = 100

num_best_select = 100

sig_decay = 0.75

start_eng = 7.5
normal_eng = 1.5
health_cap = 10
variation = 0.05

# active constants
active_eng = 0.25
active_score = 5
active_density = 10
active_signal = 0

light_decay = 0.5

light_threshold = 0.5

test = True

"""FORMAT:
DNA: [region, cell, in<con, eng, sig>, out<eng, sig, lgt>]
Healths: [region, cell]
data: [typ<sig, lgt>, region]
weights: [typ<eng, sig, lgt>, region, cell]"""


# initialize arrays, save time on memory allocation
dna = np.empty([num_regions, cells_per_region, 3, 3], dtype = np.float64)

healths = np.empty([num_regions, cells_per_region], dtype = np.float64)
region_data = np.empty([2, num_regions], dtype = np.float64)

weights = np.empty([3, num_regions, cells_per_region], dtype = np.float64)
scores = np.empty([num_regions, cells_per_region], dtype = np.float64)

rng = np.random.default_rng()

best_dna = rng.random([num_best_select, 3, 3], dtype = np.float64)

light_log = np.empty([iterations])

def randomly_distribute():
	global dna
	dna_selection = rng.integers(0, num_best_select, [num_regions, cells_per_region])
	dna[...] = best_dna[dna_selection]

	 # introduce variation
	dna += rng.random(dna.shape, dtype = np.float64) * 2 * variation
	dna -= variation

	# bound
	dna[dna < 0] = 0
	dna[dna > 1] = 1

def select_best():
	for i in range(num_best_select):
		pos = np.unravel_index(scores.argmax(), [num_regions, cells_per_region])
		scores[pos] = -1
		best_dna[i] = dna[pos]

def run_sim(active):
	global healths, selection, lit_regions, weights, scores

	# initialize region data
	healths[...] = start_eng
	region_data[0] = 0

	if active:
		region_data[1] = 0

	for round_num in range(rounds_per_iteration):

		if active:
			region_data[0] += active_signal * cells_per_region

		mult = active_density if active else 1

		# calculate weights
		for i in range(3):
			weights[i] = (
				dna[..., 0, i] + 
				dna[..., 1, i] * healths / health_cap + 
				dna[..., 2, i] * region_data[1, :, None] * mult / cells_per_region
			)
		weights[weights <= 0] = 0.001
		weights /= np.sum(weights, axis = 0)
		
		# do the things
		alive = healths < 0
		healths += normal_eng * weights[0] * alive
		region_data[0] += np.sum(weights[1] * alive, axis = 1)

		if active:
			# add light
			region_data[1] += np.sum(weights[2], axis = 1)

			# provide benefits to lit regions - note that EVERYONE benefits
			lit_regions = (region_data[1] > light_threshold * cells_per_region)[..., None]
			healths += active_eng * alive * lit_regions
			scores += active_score * alive * lit_regions

		healths -= 1
		healths[healths > health_cap] = health_cap

		# update region resources
		region_data[0] *= sig_decay
		if active:
			region_data[1] *= light_decay

def run_main():
	global dna, scores
	for iteration_num in tqdm(range(iterations)):
		# fill DNAs
		randomly_distribute()

		run_sim(False)

		np.copyto(scores, healths)

		if test:
			run_sim(True)

			# Summing scores and finding who did best
			scores += healths
			
		select_best()

		light_log[iteration_num] = np.sum(region_data[1] > light_threshold * cells_per_region * light_decay)

reshaped = dna.reshape(num_regions * cells_per_region, 3, 3)

def display_data_local():

	bins = [i / 200 for i in range(200)]

	fig, axes = plt.subplots(3, 3)

	for i in range(3):
		for j in range(3):
			ax = axes[j, i]
			ax.hist(reshaped[:, i, j], bins = bins)
			ax.axvline(reshaped[:, i, j].mean(), color = 'k', linestyle = 'dashed', linewidth = 1)
			ax.set_title(["con", "eng", "sig"][i] + " -> " + ["eng", "sig", "lgt"][j])

	fig.tight_layout()

detail = 10
test_data_arr = np.empty([2, detail, detail, 3])

def display_data_global():
	from matplotlib import cm

	for i in range(detail):
		for j in range(detail):
			test_weights = np.empty([3, reshaped.shape[0]])
			for k in range(3):
				test_weights[k] = (
					reshaped[..., 0, k] + 
					reshaped[..., 1, k] * i / detail + 
					reshaped[..., 2, k] * j / detail
				)
			
				test_data_arr[0, i, j, k] = np.mean(test_weights[k])
			test_weights[test_weights <= 0] = 0.001
			test_weights /= np.sum(test_weights, axis = 0)

			for k in range(3):
				test_data_arr[1, i, j, k] = 100 * np.sum(test_weights[k]) / reshaped.shape[0]

	fig, axes = plt.subplots(2, 3, subplot_kw = {'projection':'3d'})

	for typ in range(3):
		x, y = np.meshgrid(np.linspace(0, 1, detail), np.linspace(0, 1, detail))

		ax = axes[0, typ]

		ax.plot_surface(x, y, test_data_arr[0, ..., typ], rstride = 1, cstride = 1, cmap = cm.coolwarm)
		ax.set_xlabel("sig")
		ax.set_ylabel("eng")
		ax.set_title(["eng", "sig", "lgt"][typ])
		ax.set_zlim(-3, 3)

		ax = axes[1, typ]
		
		ax.plot_surface(x, y, test_data_arr[1, ..., typ], rstride = 1, cstride = 1, cmap = cm.coolwarm)
		ax.set_xlabel("sig")
		ax.set_ylabel("eng")
		ax.set_title(["eng", "sig", "lgt"][typ])
		ax.set_zlim(0, 100)
	
	fig.tight_layout(pad = 3)

def display_data_lightlog():
	fig, ax = plt.subplots(1)
	ax.plot(light_log * 100 / num_regions)
	ax.set_ylim(0, 100)
	ax.set_xlabel("Generation #")
	ax.set_ylabel("Percentage of lit regions")


run_main()

display_data_local()

display_data_global()

display_data_lightlog()

plt.show()








