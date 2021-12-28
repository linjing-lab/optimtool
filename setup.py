from distutils.core import setup
 
setup(
      name='optimtool',  # 包名
      version='1.0.0',  # 版本号
      description="Tools for Mathematical Optimization Region.",
      author='林景',
      author_email='1439313331@qq.com',
      url="https://gitcode.net/linjing_zyq/optimtool",
      install_requires=["numpy", "sympy", "matplotlib"],
      license='MIT License',
      platforms=["all"],
      project_urls={
        "Bug Tracker": "https://gitcode.net/linjing_zyq/optimtool/-/issues",
      },
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python',
          'Topic :: Software Development :: Libraries'
      ],
      packages=["optimtool", "optimtool.unconstrain", "optimtool.constrain", "optimtool.hybrid", "optimtool.example"], 
      python_requires=">=3.6",
)