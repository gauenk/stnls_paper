Writing an Experiment File

"groupX": Each group is meshed across the fields within themselves. For example

group0:
    foo:
    - This
    - That
    bar:
    - Bip
    - Bap

produces. [{"foo":This,"bar":Bip},{"foo":That,"bar":Bap}]

But if we want to mesh we can write

group0:
    foo:
    -
      - This0
      - This1
    - That
    bar:
    - Bip
    - Bap

producing [{"foo":This0,"bar":Bip},{"foo":This1,"bar":Bip},{"foo":That,"bar":Bap}]