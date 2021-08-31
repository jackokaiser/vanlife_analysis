from setuptools import setup, find_packages

setup(name='vanlife_analysis', version='0.0', packages=find_packages(),
      scripts=[
          'scripts/plot_weather.py',
          'scripts/plot_expenses.py',
          'scripts/plot_fuel.py'
])
