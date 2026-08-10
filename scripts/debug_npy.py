import numpy as np

mask = np.load(r'E:\lxx\V4\dataset\frl_apartment_5_mul\frl_apartment_5\bev_masks\000000.npy')
print('Shape:', mask.shape)
print('Sum per channel:', [mask[i].sum() for i in range(mask.shape[0])])

# Check argmax result
class_map = np.argmax(mask, axis=0)
print('Class map shape:', class_map.shape)
print('Unique classes:', np.unique(class_map))
print('Class distribution:')
for c in range(6):
    print(f'  Class {c}: {(class_map == c).sum()} pixels')

# Check a specific pixel
print('mask[:, 75, 75]:', mask[:, 75, 75])
print('mask[:, 0, 0]:', mask[:, 0, 0])
