"""Tests for src/utils/fcs_identities.py identity minting and school agreement."""

import pandas as pd

from src.utils.fcs_identities import (
    SYN_BASE, abbrev_match, build_fcs_identities, is_fcs_identity, schools_agree,
)


def test_abbrev_match():
    assert abbrev_match("N DAK ST", "North Dakota St")       # nflverse spelling
    assert abbrev_match("CHATTNOOGA", "Chattanooga")
    assert not abbrev_match("MAINE", "Miami")
    assert not abbrev_match("ST", "South Dakota State")      # too short


def test_schools_agree_either_direction():
    assert schools_agree(["north dakota st"], "North Dakota State", "ndakst")
    assert schools_agree(["chattanooga"], "CHATTNOOGA", "chattnooga")
    assert not schools_agree(["montana"], "Idaho", "idaho")
    assert not schools_agree([], "Idaho", "idaho")


def test_is_fcs_identity():
    mask = is_fcs_identity(pd.Series(["4241984", "-1027003", str(SYN_BASE + 7), None]))
    assert mask.tolist() == [False, False, True, False]


def _setup(tmp_path):
    pff, raw = tmp_path / "pff", tmp_path / "raw"
    for y in (2013, 2018, 2019):
        (pff / str(y)).mkdir(parents=True)
    raw.mkdir()
    fcs = pd.DataFrame([
        ("Cole Strange", 11, "CHATTNOOGA", "G"),
        ("Ben Matched", 12, "CHATTNOOGA", "G"),
    ], columns=["player", "player_id", "team_name", "position"])
    fcs.to_csv(pff / "2019" / "offense_blocking_fcs.csv", index=False)
    fcs.to_csv(pff / "2018" / "offense_blocking_fcs.csv", index=False)
    fcs.iloc[:1].assign(player="Old Timer", player_id=13).to_csv(
        pff / "2013" / "offense_blocking_fcs.csv", index=False)
    # Ben Matched's CFBD record teaches the vote map that chattnooga is
    # CFBD's "Chattanooga" (>= 3 votes needed).
    cfbd = pd.DataFrame([("77", "Ben Matched", "Chattanooga", y) for y in (2016, 2017, 2018, 2019)],
                        columns=["player_id", "player_name", "school", "season_year"])
    cfbd.to_parquet(raw / "player_gamelogs.parquet", index=False)
    grades = pd.DataFrame({
        "pff_player_id": [11, 11, 12, 12, 12, 12, 13, 14],
        "season_year":   [2018, 2019, 2016, 2017, 2018, 2019, 2013, 2019],
        "school_norm":   ["chattnooga"] * 7 + ["alabama"],
        "pff_position":  ["G", "G", "G", "G", "G", "G", "G", "WR"],
        "player_id":     [None, None, "77", "77", "77", "77", None, None],
        "match_method":  ["unmatched"] * 2 + ["exact"] * 4 + ["unmatched"] * 2,
    })
    return grades, pff, raw


def test_build_fcs_identities(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    out, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    syn = str(SYN_BASE + 11)
    # Only Cole Strange: Ben matched CFBD, Old Timer's last season is before
    # FCS coverage, pff id 14's last season isn't in an FCS file.
    assert set(rosters["player_id"]) == {syn}
    assert rosters["season_year"].tolist() == [2018, 2019]
    assert set(rosters["school"]) == {"Chattanooga"}
    assert set(rosters["position"]) == {"OG"}
    assert set(rosters["player_name"]) == {"Cole Strange"}
    assert out.loc[out["pff_player_id"] == 11, "player_id"].tolist() == [syn, syn]
    assert (out.loc[out["pff_player_id"] == 11, "match_method"] == "fcs_identity").all()
    assert out.loc[out["pff_player_id"] != 11, "player_id"].tolist() == \
        grades.loc[grades["pff_player_id"] != 11, "player_id"].tolist()


def test_name_variant_of_cfbd_player_attaches_instead_of_minting(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    # CFBD has the same Chattanooga lineman as "Coleman Strange", gamelogs only.
    pd.DataFrame([("77", "Ben Matched", "Chattanooga", y) for y in (2016, 2017, 2018, 2019)]
                 + [("88", "Coleman Strange", "Chattanooga", 2019)],
                 columns=["player_id", "player_name", "school", "season_year"]) \
      .to_parquet(raw / "player_gamelogs.parquet", index=False)
    out, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    assert set(rosters["player_id"]) == {"88"}        # seasons kept, under the CFBD id
    rows = out[out["pff_player_id"] == 11]
    assert rows["player_id"].tolist() == ["88", "88"]
    assert (rows["match_method"] == "fcs_cfbd_variant").all()


def test_transfer_down_from_fbs_attaches_to_cfbd_id(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    gl = pd.read_parquet(raw / "player_gamelogs.parquet")
    # Cole Strange at Ohio State through 2016, then Chattanooga 2018-19 in PFF.
    # A namesake whose career ends the year the FCS one starts doesn't count.
    pd.concat([gl, pd.DataFrame([("99", "Cole Strange", "Ohio State", 2015),
                                 ("99", "Cole Strange", "Ohio State", 2016)],
                                columns=gl.columns)]) \
      .to_parquet(raw / "player_gamelogs.parquet", index=False)
    out, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    assert set(rosters["player_id"]) == {"99"}
    assert rosters["school"].tolist() == ["Chattanooga", "Chattanooga"]
    assert (out.loc[out["pff_player_id"] == 11, "match_method"] == "fcs_cfbd_transfer").all()


def test_transfer_needs_a_unique_exact_name(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    gl = pd.read_parquet(raw / "player_gamelogs.parquet")
    pd.concat([gl, pd.DataFrame([("99", "Cole Strange", "Ohio State", 2016),
                                 ("98", "Cole Strange", "Iowa", 2017),
                                 ("97", "Colby Strange", "Utah", 2017)],
                                columns=gl.columns)]) \
      .to_parquet(raw / "player_gamelogs.parquet", index=False)
    _, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    assert set(rosters["player_id"]) == {str(SYN_BASE + 11)}


def test_transfer_needs_a_compatible_position(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    gl = pd.read_parquet(raw / "player_gamelogs.parquet")
    pd.concat([gl, pd.DataFrame([("99", "Cole Strange", "Ohio State", 2016)], columns=gl.columns)]) \
      .to_parquet(raw / "player_gamelogs.parquet", index=False)
    pd.DataFrame([("99", "Cole Strange", "Ohio State", 2016, "CB")],
                 columns=["player_id", "player_name", "school", "season_year", "position"]) \
      .to_parquet(raw / "rosters.parquet", index=False)
    _, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    assert set(rosters["player_id"]) == {str(SYN_BASE + 11)}     # a CB isn't the guard


def test_hyphenated_surname_is_a_variant(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    gl = pd.read_parquet(raw / "player_gamelogs.parquet")
    pd.concat([gl, pd.DataFrame([("66", "Cole Strange-Smith", "Chattanooga", 2018)], columns=gl.columns)]) \
      .to_parquet(raw / "player_gamelogs.parquet", index=False)
    out, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    assert set(rosters["player_id"]) == {"66"}
    assert set(rosters["player_name"]) == {"Cole Strange-Smith"}   # CFBD's spelling
    assert (out.loc[out["pff_player_id"] == 11, "match_method"] == "fcs_cfbd_variant").all()


def test_fbs_only_candidate_is_not_minted(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    # CFBD has a Chattanooga roster both seasons, so Cole Strange's rows are
    # an unmatched FBS player, not an FCS one.
    pd.DataFrame([("55", "Someone Else", "Chattanooga", y, "QB") for y in (2018, 2019)],
                 columns=["player_id", "player_name", "school", "season_year", "position"]) \
      .to_parquet(raw / "rosters.parquet", index=False)
    out, rosters = build_fcs_identities(grades, pff, raw, board_path=tmp_path / "none.csv")
    assert rosters.empty
    assert out.loc[out["pff_player_id"] == 11, "player_id"].isna().all()
    assert (out.loc[out["pff_player_id"] == 11, "match_method"] == "unmatched").all()


def test_transfer_rejected_when_cfbd_player_was_drafted_first(tmp_path):
    grades, pff, raw = _setup(tmp_path)
    gl = pd.read_parquet(raw / "player_gamelogs.parquet")
    pd.concat([gl, pd.DataFrame([("99", "Cole Strange", "Ohio State", 2016)], columns=gl.columns)]) \
      .to_parquet(raw / "player_gamelogs.parquet", index=False)
    board = tmp_path / "board.csv"
    pd.DataFrame([("Cole Strange", "Ohio State", 2017)], columns=["name", "school", "year"]).to_csv(board, index=False)
    _, rosters = build_fcs_identities(grades, pff, raw, board_path=board)
    assert set(rosters["player_id"]) == {str(SYN_BASE + 11)}
