import re
import sys

from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'optimtool-2.4.4 requires Python >=3.7, but yours is {sys.version}')

if (3, 7, 0) <= sys.version_info < (3, 8, 0):
    # https://github.com/pypa/setuptools/issues/926#issuecomment-294369342
    try:
        import fastentrypoints
    except ImportError:
        try:
            import pkg_resources
            from setuptools.command import easy_install

            easy_install.main(['fastentrypoints'])
            pkg_resources.require('fastentrypoints')
        except:
            pass

VERSION_FILE = 'optimtool/_version.py'
VERSION_REGEXP = r'^__version__ = \'(\d+\.\d+\.\d+)\''

r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
    raise RuntimeError(f'Unable to find version string in {VERSION_FILE}.')

version = r.group(1)

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''
 
setup(
      name='optimtool',  # pkg_name
      packages=[
          "optimtool",
          "optimtool.unconstrain",
          "optimtool.constrain",
          "optimtool.hybrid",
          "optimtool.example"
      ],
      version=version,  # version number
      description="The fundamental package for scientific research in optimization.",
      author='林景',
      author_email='linjing010729@163.com',
      license='MIT',
      url='https://github.com/linjing-lab/optimtool',
      download_url='https://github.com/linjing-lab/optimtool/tags',
      long_description=_long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      zip_safe=False,
      setup_requires=['setuptools>=18.0', 'wheel'],
      project_urls={
            'Source': 'https://github.com/linjing-lab/optimtool/tree/master/optimtool/',
            'Tracker': 'https://github.com/linjing-lab/optimtool/issues',
      },
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires = [
            'numpy>=1.21.0', 
            'sympy>=1.9',
            'matplotlib>=3.2.0'
      ],
      # extras_require = []
)
