from setuptools import setup

setup(
    name='rex github',
    version='1.0',
    packages=['rex', 'test', 'test.check_tests', 'test.model_tests', 'test.preprocessing_tests',
              'test.model_selection_tests'],
    package_dir={'': 'src'},
    package_data={'src': ['/*']},
    url='',
    license='',
    author='Lorenzo',
    author_email='l.gardo98@gmail.com',
    description='',
    install_requires=['numpy',
                      'surprise'
                      'pandas',
                      'requests',
                      'fastapi',
                      'lightfm',
                      'scikit-learn',
                      'matplotlib',
                      'seaborn',
                      'scipy',
                      'setuptools']
)
