import numpy as np 
import h5py
import os 








if __name__ == "__main__":
    
    event_path = '/mnt/workspace/ywb/WorldNew/rpg_vid2e_v2/rpg_vid2e/outputs/event/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/gt_0/'
    npz_idx = "0000000000.npz"
    
    npz_path = os.path.join(event_path,npz_idx)
    e = np.load(npz_path)
    
    print(f"e:{e}")