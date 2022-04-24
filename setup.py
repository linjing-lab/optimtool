from setuptools import setup
 
setup(
      name='optimtool',  # 包名
      version='2.3.5',  # 版本号
      description="Tools for Mathematical Optimization Region.",
      long_description = "Simplify how data is constructed outside the library",
      author='林景',
      author_email='1439313331@qq.com',
      url="https://github.com/linjing-lab/optimtool",
      packages=[
          "optimtool",
          "optimtool.unconstrain",
          "optimtool.constrain",
          "optimtool.hybrid",
          "optimtool.example",
          "optimtool.functions"
      ],
      include_package_data=True,    # 自动打包文件夹内所有数据
      zip_safe=True, # 设定项目包为安全，不用每次都检测其安全性
      install_requires = [
            'numpy',
            'sympy',
            'matplotlib'
      ],
)
