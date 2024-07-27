PhotoPrism: Computer Vision Models
==================================

Supplementary computer vision models that can be used with PhotoPrism.

## Installing Build Dependencies

Please make sure that you have Git and Python 3 installed on your system. On Ubuntu/Debian Linux you can run the following command for this:

```
sudo apt-get install -y git python3 python3-pip python3-venv python3-wheel
```

You can then proceed with the installation of the Python dependencies in a virtual environment by either using the `Makefiles` (run `make` in the main project directory) or by manually running the following commands in the respective subdirectory, e.g.:

```bash
cd describe
python3 -m venv venv
. ./venv/bin/activate
./venv/bin/pip3 install --disable-pip-version-check -r requirements.txt
```

## Submitting Pull Requests

Follow our [step-by-step guide](https://docs.photoprism.app/developer-guide/pull-requests) to learn how to submit new features, bug fixes, and documentation enhancements.

----

*PhotoPrism® is a [registered trademark](https://www.photoprism.app/trademark). By using the software and services we provide, you agree to our [Terms of Service](https://www.photoprism.app/terms), [Privacy Policy](https://www.photoprism.app/privacy), and [Code of Conduct](https://www.photoprism.app/code-of-conduct). Docs are [available](https://link.photoprism.app/github-docs) under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/); [additional terms](https://github.com/photoprism/photoprism/blob/develop/assets/README.md) may apply.*
