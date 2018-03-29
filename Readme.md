# Sentiment modeling

Wikipedia's talk page is where editors discuss changes and improvements to articles on Wikipedia. Editors can sometimes use toxic languages in their discussion. This code predicts the different types of toxicity like threats, obscenity, insults and identity-based hate in their comments. It uses two bidirectional GRU layers, followed by a dense layer. Words in the comments are represented by the frozen word embedding from FastText trained on Common Crawl data.
