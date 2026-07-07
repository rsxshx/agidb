# agidb CLI — the `agidb` wrapper

There is no `agidb-server` process. The Rust binary is a single
file-IO client — every command opens the on-disk store, does the
work, exits. To make this feel like a real database CLI (the way
`psql` feels like a real SQL client, not just a wrapper around
`libpq`), ship a small `agidb` shell wrapper.

## Install (one-time)

```bash
mkdir -p ~/bin ~/.config/agidb ~/.local/share/agidb
# build the rust binary
cargo build --release -p agidb-cli
# install the wrapper
cp docs/cli/agidb ~/bin/agidb
chmod +x ~/bin/agidb
# add to PATH (if not already)
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
```

## Concepts

- **dbs** — named stores under `~/.local/share/agidb/<name>/`.
  `default` is auto-created by `agidb init`.
- **current** — a pointer at `~/.config/agidb/current` whose contents
  are the active db name. `agidb use <name>` flips it.
- **`AGIDB_DB` env var** — overrides the current pointer for one
  command (or `export AGIDB_DB=...` for a session). Useful for CI or
  pointing at a per-project store.

## Usage

```bash
agidb init                                  # create ~/.local/share/agidb/default
agidb use dev                              # switch to a db called "dev" (creates it)
agidb current                              # show active db name
agidb where                                # show active db path

agidb observe "Sarah recommended Bawri"    # add a memory (--offline for text-only)
agidb recall "what thai place?"            # query
agidb get 3                                # show episode 3
agidb list                                 # show recent episodes
agidb stats                                # counts
agidb sense "novel signal"                 # surprise-gated sensory frame
agidb sensory                              # recent sensory frames
agidb consolidate                          # one consolidation pass

agidb goal-set "find a thai place for dinner"
agidb goal-list
agidb belief-assert "Sarah likes thai food"
agidb belief-list

agidb serve                                # MCP stdio server (for Claude/Cursor)

agidb dbs                                  # list known db names
agidb reset                                # wipe the active db (interactive)
```

## Per-project stores

The standard pattern is one db per project / per machine. Pointing
at a project-local store from a script:

```bash
AGIDB_DB=.agidb/memory agidb observe "..."   # hidden dir inside the project
AGIDB_DB=.agidb/memory agidb recall "..."
```

This is the equivalent of `DATABASE_URL` for Postgres.

## Reset

```bash
agidb reset                                # wipes the active db after confirmation
```

## Where data lives

- `~/.config/agidb/current`     — one-line file: the active db name
- `~/.config/agidb/known`       — list of db names `use` has ever pointed at
- `~/.local/share/agidb/<name>/meta.redb`     — redb metadata file
- `~/.local/share/agidb/<name>/signatures.dat` — mmap'd HV file

## Why not just call `target/release/agidb` directly?

You can — `target/release/agidb observe <db-path> "<text>" --offline`
works. The wrapper exists so:

1. You don't pass the path every time (analogous to `psql` knowing
   which database to connect to).
2. The db path is a *name*, not a file system path — switch between
   projects with `agidb use work` / `agidb use dev`.
3. The rust binary may move (rebuilds, releases); the wrapper points
   at `$AGIDB_BIN` and can be redirected without editing scripts.
4. Future helpers (shell completion, db listing, reset, format
   conversions) live in one place.
