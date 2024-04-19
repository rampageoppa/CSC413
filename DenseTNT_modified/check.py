import pickle

with open('/home/guxunjia/project/DenseTNT_modified/bev_results/result.pkl', 'rb') as handle:
    data = pickle.load(handle)

breakpoint()


# Paths to your pickle files
pickle_files = ['/home/guxunjia/project/DenseTNT_modified/bev_results/traj_batch_9.pkl', 
'/home/guxunjia/project/DenseTNT_modified/bev_results/traj_batch_6.pkl', '/home/guxunjia/project/DenseTNT_modified/bev_results/traj_batch_7.pkl', 
'/home/guxunjia/project/DenseTNT_modified/bev_results/traj_batch_8.pkl']

# Initialize an empty dictionary to hold the merged results
merged_dict = {}

# Iterate through the list of pickle files
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as file:
        # Load the dictionary from the current pickle file
        current_dict = pickle.load(file)
        
        # Merge it with the merged_dict
        # This will update merged_dict with the items from current_dict
        merged_dict.update(current_dict)

# At this point, merged_dict contains the merged content from all pickle files
# You can now save merged_dict back to a pickle file if needed
with open('/home/guxunjia/project/DenseTNT_modified/bev_results/merged_dict.pkl', 'wb') as output_file:
    pickle.dump(merged_dict, output_file)
breakpoint()
print("Merging completed and saved to 'merged_dict.pickle'.")
