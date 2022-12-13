from oml.dali.dataset import download_dali_if_necessary, DALI_DATASET_DIR

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.backend_impl import TensorListGPU, TensorListCPU

image_dir = str(DALI_DATASET_DIR)
max_batch_size = 4


def show_images(image_batch):
    if isinstance(image_batch, TensorListGPU):
        image_batch = image_batch.as_cpu()
    columns = 4
    rows = (max_batch_size + 1) // (columns)
    fig = plt.figure(figsize = (12,(12 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch[j])
    plt.show()


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='mixed')

    width = 200
    height = 200

    images = fn.resize(images, size=[height, width], mode="not_larger")
    images = fn.crop(images, crop_w=width, crop_h=height, out_of_bounds_policy="pad")

    return images, labels


download_dali_if_necessary()


pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
pipe.build()
pipe_out = pipe.run()
images, labels = pipe_out
show_images(images)






