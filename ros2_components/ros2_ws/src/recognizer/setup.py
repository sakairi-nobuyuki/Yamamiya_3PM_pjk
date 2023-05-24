from setuptools import setup

package_name = "recognizer"

setup(
    name=package_name,
    version="0.0.0",
    packages=[
        package_name,
        "recognizer/components/streamer",
        "recognizer/components/region_extractor",
        "recognizer/components/region_extractor/train",
        "recognizer/io",
        "recognizer/pipelines/data_collection",
        "recognizer/pipelines/train",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="nsakairi",
    maintainer_email="SAKAIRI.Nobuyuki@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "image_streaming_node = recognizer.image_streaming_node:main",
            "thresholding_train_node = recognizer.thresholding_train_node:main",
            "object_filter_train_node = recognizer.object_filter_train_node:main",
        ],
    },
)
