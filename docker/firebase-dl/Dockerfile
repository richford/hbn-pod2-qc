FROM node:16-alpine

RUN apk add bash

COPY dl_firebase_project.sh /usr/local/bin/dl_firebase_project.sh
RUN chmod +x /usr/local/bin/dl_firebase_project.sh 

RUN npm install -g firebase-tools
EXPOSE 9005

ENTRYPOINT ["dl_firebase_project.sh"]