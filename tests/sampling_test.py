from ivt.sampling import sample_sphere
from matplotlib import pyplot as plt 
from pathlib import Path

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.\n')
    tests.append(wrapper)

@add_test 
def sphere_sampling():
    n = 100
    radius = 2
    methods = ['uniform', 'stratified', 'fibonacci']
    fig = plt.figure()

    output_path = Path('tmp_output', 'sampling_test', 'sphere_sampling')
    output_path.mkdir(parents=True, exist_ok=True)

    for method in methods:
        print(f'Testing {method}...')
        points = sample_sphere(n, radius, method)
        ax = fig.add_subplot(projection='3d')
        ax.set_title(method)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        plt.savefig(output_path / f'{method}.png')
        plt.clf()

if __name__ == '__main__':
    for test in tests:
        test()