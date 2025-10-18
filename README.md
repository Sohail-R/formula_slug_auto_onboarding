# formula_slug_auto_onboarding
These are my projects for Formula Slug's software automation team onboarding.


First of all, I learned the Pytorch material from learnpytorch.io and the day long youtube video that is associated with it, and I wrote down all the exercises associated with the 00 section.
My project is focused on a neural network that predicts the rarity of Clash Royale cards. 

This was built off of a Feedforward Neural Network and used Pytorch and scikit-learn to train the model and do predictions. Initially, the vision was to have the model predict the cards themselves, but the dataset that I used didn't have enough variety in the information that was provided, as there was only one set of information per card. This led to the model basically "guessing" for every card, and thus I changed the prediction to the card rarity instead, which had a much better outcome.

Additionally, I used matplotlib to graph the loss per epoch.
