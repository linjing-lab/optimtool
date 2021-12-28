from distutils.core import setup
 
setup(
      name='optimtool',  # 包名
      version='2.2.2',  # 版本号
      description="Tools for Mathematical Optimization Region.",
      author='林景',
      author_email='1439313331@qq.com',
      url="https://gitcode.net/linjing_zyq/optimtool",
      packages=["optimtool", "optimtool.unconstrain", "optimtool.constrain", "optimtool.hybrid", "optimtool.example", "functions"], 
      install_requires=['numPy', 'matplotlib', 'sympy'],
)
