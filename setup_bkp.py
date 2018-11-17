from setuptools import setup, find_packages

setup(
      name='sentimentanalyser',
      version='1.5',
      description='',
      url='',
      author='',
      author_email='',
      keywords='',
      license='',
      packages=['sentimentanalyser'],
      install_requires=[
            "et-xmlfile==1.0.1",
            "jdcal==1.4",
            "nltk==3.3",
            "numpy==1.15.3",
            "openpyxl==2.5.9",
            "pandas==0.23.4",
            "psycopg2==2.7.5",
            "python-dateutil==2.7.5",
            "pytz==2018.7",
            "scikit-learn==0.20.0",
            "scipy==1.1.0",
            "six==1.10.0",
            "whitenoise==4.1",
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False
)