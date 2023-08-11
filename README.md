# Launchpad Spring 2023 Semester Project
## lofi bytes API

By: Alicia Wang [PL], Alena Chao, Eric Liu, Zane Mogannam, Chloe Wong, Iris Zhou

## So, what is lofi bytes?
Picture this: it is midterm season, you have a ton of work to finish, and you need a long library grind session. You reach for your headphones, but instead of turning on Spotify, you open lofi bytes: an aesthetic web app that takes song samples and outputs chill lofi, complete with customizable background sounds and beats.

Over the Spring 2023 semester, our team has been creating an integrated, user-friendly web application that allows users to generate lofi tracks from input MIDI samples and customize further with sounds of rain, fire, and cafe ambiance. This article will outline our process from education to the final product, discuss everything we did, from training an ML model to building a full-stack application, and reflect on limitations, extensions, and further learning opportunities.

Check out our website at https://callaunchpad.github.io/lofi-bytes-app/! This is a semester project by Launchpad, a creative ML organization founded on the UC Berkeley campus.

## About our repo

Our team used the Flask API to deploy our ML model. Flask is a lightweight, easy-to-use web framework allowing developers to build and deploy web applications quickly. By leveraging Flask’s routing and request-handling capabilities, we can easily expose our trained ML model as an endpoint that can accept incoming data and return predictions in real time. This makes it possible to integrate our model into other applications, such as mobile apps, chatbots, and in this case, our lofi bytes web application!

When users upload a MIDI file through our Lofi-Bytes web app, an Axios post request sends the data to the ML Model (which resides in the Flask API). Our model generates an output MIDI, returns it to our React front-end, and the generated lofi music is played for our users!

Check out our models at https://github.com/callaunchpad/lofi-bytes.

Our web app code is located at https://github.com/callaunchpad/lofi-bytes-app.

