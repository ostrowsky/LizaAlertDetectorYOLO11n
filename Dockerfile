FROM python:3.10

RUN useradd -m user
ENV HOME=/home/user
ENV PATH="$HOME/.local/bin:$PATH"
WORKDIR $HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python3", "app.py"]
