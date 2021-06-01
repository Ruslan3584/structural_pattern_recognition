import argparse
import numpy as np
import matplotlib.pyplot as plt

def xor_columns(col1,col2, noise_level):
    # xor 2 columns
    return np.sum((col1^col2)*np.log(noise_level) + (1^col1^col2)*np.log(1-noise_level))

def get_equation(input_string, dataset, noise_level):
    # create image from input string 
    input_string = input_string.replace('#','2')
    
    input_array = np.array(list(map(list,input_string.split(" ")))).ravel().astype(int)
    n_rows = 3
    n_cols = int(input_array.shape[0]/n_rows)
    pic = np.hstack(dataset[input_array])
    # apply noise
    input_equation = np.vstack(np.hsplit(pic,n_rows))
    ksi = np.random.binomial(size=input_equation.size, n=1, p=noise_level).reshape(input_equation.shape)
    noised_image = ksi^input_equation
    noised_image_splitted = np.hsplit(noised_image,n_cols)
    # reference columns(column as a label |K| = 8)
    reference_symbols = np.array([0,0,0,1,1,1,0,1,
                                  0,0,1,0,1,0,1,1,
                                  0,1,0,0,0,1,1,1])
    n_rows_ref = 3
    n_cols_ref = int(reference_symbols.shape[0]/n_rows_ref)
    pic = np.hstack(dataset[reference_symbols])
    reference_images = np.array(np.hsplit(np.vstack(np.hsplit(pic,n_rows_ref)),n_cols_ref))
    
    return input_equation, noised_image, reference_images

def get_best_paths(reference_images,noised_image, noise_level):
    n_cols = int((noised_image.shape[1]/noised_image.shape[0])*3)
    noised_image_splitted = np.hsplit(noised_image,n_cols)
    result = []
    # -inf if combination position-label is impossible
    f = np.full((n_cols,8,2),-np.inf)
    for c in [0,5,6]:
        f[-1,c,0] = xor_columns(reference_images[c],noised_image_splitted[-1], noise_level)
    for c in [4]:
        f[-1,c,1] = xor_columns(reference_images[c],noised_image_splitted[-1], noise_level)
    for column in range(n_cols-2,-1,-1):
        for c in [0,5,6]:
            f[column,c,0] = xor_columns(reference_images[c],noised_image_splitted[column], noise_level) + np.max(f[column+1,:,0])
        for c in [1]:
            f[column,c,0] = xor_columns(reference_images[c],noised_image_splitted[column], noise_level) + np.max(f[column+1,:,1])


        for c in [4]:
            f[column,c,1] = xor_columns(reference_images[c],noised_image_splitted[column], noise_level) + np.max(f[column+1,:,0])
        for c in [2,3,7]:
            f[column,c,1] = xor_columns(reference_images[c],noised_image_splitted[column], noise_level) + np.max(f[column+1,:,1])
        result.append(np.argmax(np.max(f[column+1,:,:],axis=1)))
    result.append(np.argmax(np.max(f[0,:,:],axis=1)))
    result = result[::-1]
    # create image from best labeling(result)
    result_img = np.hstack(reference_images[result])
    return result_img


def main():

    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", type=str,   help="first summand second summand solution. Use '#' for space!")
    parser.add_argument("noise_level", type=float,   help="noise level for Bernoulli distribution")
    args = parser.parse_args()

    dataset = np.load('terminal_symbols.npy')
    input_equation_image, noised_image, reference_images = get_equation(args.input_string, dataset, args.noise_level)

    result_img = get_best_paths(reference_images, noised_image, args.noise_level )

    plt.imsave("input_equation.png", input_equation_image, cmap='binary')
    plt.imsave("noised_image.png",   noised_image, cmap='binary')
    plt.imsave("solution.png", result_img, cmap='binary')


if __name__ == "__main__":
    main()
