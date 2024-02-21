import demo_functions
import demo_images

def prepare_demos(precompute_size):
    demo_functions.save_demos(precompute_size)
    demo_images.save_demos(precompute_size)

if __name__ == "__main__":
    prepare_demos(800)