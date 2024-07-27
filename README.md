PhotoPrism: Computer Vision Models
==================================

[![License: Apache 2.0](https://img.shields.io/badge/license-apache-313055.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Documentation](https://img.shields.io/badge/read-the%20docs-4d6a91.svg)](https://docs.photoprism.app/developer-guide/)
[![Community Chat](https://img.shields.io/badge/chat-on%20gitter-4d6a91.svg)](https://link.photoprism.app/chat)
[![GitHub Discussions](https://img.shields.io/badge/ask-%20on%20github-4d6a91.svg)](https://link.photoprism.app/discussions)
[![Bluesky Social](https://dl.photoprism.app/img/badges/badge-bluesky.svg)](https://bsky.app/profile/photoprism.app)
[![Mastodon](https://dl.photoprism.app/img/badges/badge-floss-social.svg)](https://floss.social/@photoprism)

The software in this repository is licensed under the Apache License, Version 2.0 (the “License”). You can obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

## Installing Build Dependencies

First of all, please make sure that you have [Git](https://git-scm.com/downloads) and [Python 3](https://www.python.org/downloads/) installed on your system, e.g. by running the following command on Ubuntu/Debian Linux:

```
sudo apt-get install -y git python3 python3-pip python3-venv python3-wheel
```

You can then install the required Python dependencies in a virtual environment by either using the `Makefiles` we provide (i.e. run `make` in the main project directory or a subdirectory) or by running the following commands in a service subdirectory, e.g:

```bash
cd describe
python3 -m venv venv
. ./venv/bin/activate
./venv/bin/pip3 install --disable-pip-version-check -r requirements.txt
```

## Submitting Pull Requests

Follow our [step-by-step guide](https://docs.photoprism.app/developer-guide/pull-requests) to learn how to submit new features, bug fixes, and documentation enhancements.

----

*Copyright © 2024 [PhotoPrism UG](https://www.photoprism.app/contact). By using the software and services we provide, you agree to our [Terms of Service](https://www.photoprism.app/terms), [Privacy Policy](https://www.photoprism.app/privacy), and [Code of Conduct](https://www.photoprism.app/code-of-conduct). PhotoPrism® is a [registered trademark](https://www.photoprism.app/trademark).*
