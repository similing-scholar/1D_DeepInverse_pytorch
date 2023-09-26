import numpy as np
import h5py
import get_raw_data as rawdata


def save_h5(data, filename):
    """h5py格式会压缩，占用空间小，读取速度快，但是不能直接用文本编辑器打开"""
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('data', data=data)
    h5f.close()


def save_csv(data, filename):
    np.savetxt(filename, data, delimiter=',')


def save_format(data, filename):
    if filename.endswith('.h5'):
        save_h5(data, filename)
        print(f'{data.shape} saved to {filename}')
    elif filename.endswith('.csv'):
        save_csv(data, filename)
        print(f'{data.shape} saved to {filename}')
    else:
        print('Unsupported file format.')


if __name__ == '__main__':
    num_vectors = 100
    vector_length = 42
    spectrum_dataset = rawdata.generate_gaussian_vectors(num_vectors, vector_length, 5)
    spectrum_filename = f'../dataset/1Dspectrum[{num_vectors},{vector_length}].h5'
    save_format(spectrum_dataset, spectrum_filename)

    # mask = rawdata.get_array_filters(42, 101)
    # mask_filename = '../dataset/mask[42,101].h5'
    # save_format(mask, mask_filename)
