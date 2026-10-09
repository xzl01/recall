# Migrating an existing installation

The default state directory is now `~/.local/share/recall`, rather than a checkout-specific directory. Existing installations remain separate unless you explicitly point the new executable at them.

To read an existing npulab index, configure `RECALL_HOME` to its state directory and `RECALL_MODEL_DIR` to the existing BGE directory. Back up **both** `index/meta.db` and `index/vectors.npy` first, along with the old executable and code. Never replace only one member of the index pair.

`recall migrate` converts legacy message-sized records into token windows while preserving all indexed corpora. An index already using windows requires no migration. The new daemon uses a different default socket and checks model identity; the older daemon can remain running independently.

New session indexing records relative paths within the configured session root. Older indexes used file basenames; nested paths may therefore be treated as new sources until `recall index sessions --force` rebuilds the selected Pi history. Confirm the session root before forcing a rebuild.

New directory corpus IDs add a root-path hash. Old `dir:project` records remain readable through the prefix filter; forcing a new hashed corpus does not delete old un-hashed records. For a clean directory migration, use a new state directory and index the desired directories, keeping the old paired index as a backup. There is no automatic destructive cleanup of legacy corpora.

The standalone package has been verified with isolated synthetic data and publicly sourced model weights. It has not replaced an existing production installation as part of repository publication.
