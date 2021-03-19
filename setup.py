from distutils.core import setup

setup(name="gmm-dimension-reduction",
      version="0.0.1",
      author="Laines Schmalwasser",
      package_dir={"": "src"},
      packages=["gmm_dimension_reduction"],
      python_requires='>=3.6',
      install_requires=["numpy", "tqdm"]
      )
