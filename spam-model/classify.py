import pickle
from spammodel import SpamModel

def classify(document):
    with open("./model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.classify(document)

if __name__ == "__main__":
  text = """
    > Hi!
    > 
    > Is there a command to insert the signature using a combination of keys and not
    > to have sent the mail to insert it then?

    I simply put it (them) into my (nmh) component files (components,
    replcomps, forwcomps and so on).  That way you get them when you are
    editing your message.  Also, by using comps files for specific
    folders you can alter your .sig per folder (and other tricks).  See
    the docs for (n)mh for all the details.

    There might (must?) also be a way to get sedit to do it, but I've
    been using gvim as my exmh message editor for a long time now.  I
    load it with a command that loads some email-specific settings, eg,
    to "syntax" colour-highlight the headers and quoted parts of an
    email)... it would be possible to map some (vim) keys that would add
    a sig (or even give a selection of sigs to choose from).

    And there are all sorts of ways to have randomly-chosen sigs...
    somewhere at rtfm.mit.edu... ok, here we go:
    rtfm.mit.edu/pub/usenet-by-group/news.answers/signature_finger_faq.
    (Warning... it's old, May 1995).

    > Regards,
    > Ulises

    Hope this helps.

    Cheers
    Tony
    """

  print(classify(text))
  