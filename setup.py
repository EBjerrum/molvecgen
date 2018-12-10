from setuptools import setup

setup(name='molvecgen',
      version='0.1',
      description='molecular vectorizer and batch generator',
      #url='',
      author='Esben Jannik Bjerrum, kfxl284',
      author_email='esben.bjerrum@astrazeneca.com',
      license='MIT',
      packages=['molvecgen'],
      install_requires=[
          #'rdkit', #not available through git but conda
          'numpy'
      ],
      zip_safe=False,
      )
