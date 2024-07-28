# INSTALLS OPTIONAL PACKAGES AND DRIVERS IN DOCKER IMAGES

export PATH="/usr/local/sbin:/usr/sbin:/sbin:/usr/local/bin:/usr/bin:/bin:/app/scripts:/root/.cargo/bin:/app/venv/bin"

apt-upgrade: update
update:
	apt-get update
	apt-get -qq dist-upgrade
apt-cleanup: clean
clean:
	apt-get -y autoremove
	apt-get -y autoclean
	rm -rf /var/lib/apt/lists/*

# Declare all targets as "PHONY", see https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html.
MAKEFLAGS += --always-make
