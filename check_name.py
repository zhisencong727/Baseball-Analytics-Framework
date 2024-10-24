from pybaseball import playerid_lookup,get_splits,playerid_reverse_lookup

player1 = playerid_lookup("Call","Alex",fuzzy=True)

dict = player1.to_dict()


print(dict)
print(next(iter(dict['key_mlbam'].values())))

dict1 = playerid_reverse_lookup([571448],key_type='mlbam').to_dict()
print(next(iter(dict1['key_bbref'].values())))
