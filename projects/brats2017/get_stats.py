stats = []

def get_size(scan, axis):
    return np.max(np.sum(scan, axis=axis))
    
for patient in tqdm(data_loader.patients):
    scan = data_loader.load_mscan(patient)[0]
    segmentation = data_loader.load_mscan(patient)
    
    bottom = scan.min()
    brain = scan != bottom
    
    brain_sizes = [get_size(brain, i) for i in range(3)]
    brain_volume = np.sum(brain)
    
    cancer = segmentation > 0
    cancer_sizes = [get_size(cancer, i) for i in range(3)]
    cancer_volume = np.sum(cancer)
    
    record = {'patient': patient,
              'brain_x_size': brain_sizes[0],
              'brain_y_size': brain_sizes[1],
              'brain_z_size': brain_sizes[2],
              'cancer_x_size': cancer_sizes[0],
              'cancer_y_size': cancer_sizes[1],
              'cancer_z_size': cancer_sizes[2],
              'brain_volume': brain_volume,
              'cancer_vilume': cancer_volume,
              'lgg_cancer': patient in data_loader.lgg_patients
             }
    stats.append(record)