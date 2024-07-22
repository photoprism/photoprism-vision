FROM python:3.13.0b4-bookworm

WORKDIR /app

COPY requirements.txt ./

RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf -y | sh

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "flask", "--app", "app", "run"]