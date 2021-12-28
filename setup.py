import setuptools

setuptools.setup(
    name="optimtool",
    version="0.1.1",
    author="林景",
    author_email="1439313331@qq.com",
    description="Tools for Mathematical Optimization Region.",
    url="https://gitcode.net/linjing_zyq/optimtool",
    project_urls={
        "Bug Tracker": "https://gitcode.net/linjing_zyq/optimtool/-/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["optimtool", "optimtool.unconstrain", "optimtool.constrain", "optimtool.hybrid", "optimtool.example"], 
    python_requires=">=3.6",
)
