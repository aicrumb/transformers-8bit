import setuptools

setuptools.setup(
    name='gptj_8bit',
    version='0.0.1',
    author='aicrumb',
    author_email='crumby.naomi@gmail.com',
    description='code to load gpt-j-6b in 8bit',
    long_description='i was bored of writing the code myself every time lol',
    long_description_content_type="text/markdown",
    url='https://github.com/aicrumb/gpt-j-8bit',
    project_urls = {
        "Bug Tracker": "https://github.com/aicrumb/gpt-j-8bit/issues"
    },
    license='GNU GPLv3',
    packages=['gptj_8bit'],
    install_requires=['transformers', 'datasets'],
)