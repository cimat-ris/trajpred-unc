from setuptools import setup,find_packages

setup(
   name='trajpred-unc',
   version='1.0',
   author='Jean-Bernard Hayet',
   author_email='jbhayet@cimat.mx',
   packages=find_packages(include=['opentraj']),
   url='https://github.com/crowdbotp/OpenTraj',
   license='Apache',
   description='Tools for analyzing trajectory datasets',
   long_description=open('README.md').read(),
   install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "tqdm",
        "pykalman", 
        "PyYAML",       
   ],
   extras_require={
        'test': [
            "pylint",
            "pytest",
        ],
        'plot': [
            "matplotlib",
            "seaborn",
        ]
   }
)
