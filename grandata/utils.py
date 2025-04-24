import os
import asyncio
import zarr
import icechunk

# asynchronous conversion function.
async def _convert_zarr_store_to_icechunk(source_store, target_path):
    # Create a local Icechunk repository.
    target_storage = icechunk.local_filesystem_storage(target_path)
    repo = icechunk.Repository.create(target_storage)
    
    # Create a writable session.
    session = repo.writable_session("main")
    target_store = session.store

    # Copy all entries (metadata and data) from the source to target.
    async for key in source_store.list():
        value = await source_store.get(key)
        # Use the asynchronous setter provided by Icechunk.
        await target_store.set(key, value)
    
    # Commit the transaction.
    commit_id = session.commit("Converted Zarr store to Icechunk store")
    print("Conversion complete. Commit ID:", commit_id)
    return repo

# Synchronous helper
def convert_zarr_store_to_icechunk(source_path, target_path):
    '''convert a zarr store to an icechunk store'''
    source_store = zarr.storage.LocalStore(source_path)
    try:
        # Try getting the current event loop.
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # If a loop exists and is running, use nest_asyncio to allow nested event loops.
    if loop and loop.is_running():
        try:
            import nest_asyncio
        except ImportError as e:
            raise RuntimeError("An event loop is already running. "
                               "Install `nest_asyncio` to allow nested event loops.") from e
        nest_asyncio.apply()

    # Now run the async function in a synchronous context.
    return asyncio.run(_convert_zarr_store_to_icechunk(source_store, target_path))
