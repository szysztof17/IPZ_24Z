import asyncio
from typing import List, Callable, Awaitable

import aiomysql
from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd

from molecule_generation import create_generator_model, prepare_training_data

async def insert_qed(conn: aiomysql.Connection) -> None:
    async with conn.cursor() as cursor:
        await cursor.execute("DROP TABLE IF EXISTS existing_qed")
        await cursor.execute("""
            CREATE TABLE existing_qed (
                cid INT NOT NULL,
                qed FLOAT NULL,
                PRIMARY KEY(cid)
            )
        """)
    async with conn.cursor() as cursor:
        await cursor.execute("SELECT cid, canonicalsmiles FROM pre_filtered")
        all_smiles = await cursor.fetchall()

        for i, element in enumerate(all_smiles):
            if i % 10000 == 0:
                print(i)

            cid, smiles = element
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                qed = None
            else:
                qed = QED.qed(mol)
            await cursor.execute("INSERT INTO existing_qed VALUES (%s, %s)", (cid, qed))
    return

async def prepare_generation_source(conn: aiomysql.Connection) -> None:
    print("Preparing table for generation source")
    async with conn.cursor() as cursor:
        await cursor.execute("DROP TABLE IF EXISTS source_molecules")
        await cursor.execute("""
            CREATE TABLE source_molecules (
                id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                canonicalsmiles TEXT NULL,
                qed FLOAT NOT NULL
            )
        """)
    async with conn.cursor() as cursor:
        await cursor.execute("""
                INSERT INTO source_molecules (canonicalsmiles, qed)
                SELECT canonicalsmiles, qed FROM pre_filtered
                    LEFT JOIN existing_qed ON pre_filtered.cid = existing_qed.cid
                WHERE existing_qed.qed IS NOT NULL ORDER BY qed DESC LIMIT 200000
            """)

async def generate_new(conn: aiomysql.Connection) -> None:
    async with conn.cursor() as cursor:
        await cursor.execute("DROP TABLE IF EXISTS generated_molecules")
        await cursor.execute("""
            CREATE TABLE generated_molecules (
                id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                canonicalsmiles TEXT NOT NULL,
                qed FLOAT NOT NULL
            )
        """)
    async with conn.cursor() as cursor:
        breakpoints = range(5, 52, 2)

        max_atoms = 60

        latent_dim = 1200
        vae_lr = 1e-4
        molecules_to_generate = 50
        max_molsize = 50
        epochs = 2000

        print(f"Creating model with { {'max_molsize': max_molsize, 'vae_lr': vae_lr, 'max_atoms': max_atoms, 'latent_dim': latent_dim} }")
        model = create_generator_model(max_molsize=max_molsize, vae_lr=vae_lr, num_atoms=max_atoms,latent_dim=latent_dim)

        for limit in breakpoints:
            print(f"Fetching training data(len <= {limit})")
            await cursor.execute(f"""
                SELECT canonicalsmiles, qed FROM pre_filtered
                LEFT JOIN existing_qed ON pre_filtered.cid = existing_qed.cid and existing_qed.qed is not null
                WHERE LENGTH(canonicalsmiles) <= {limit}
                ORDER BY qed DESC LIMIT 10000
            """)
            data = await cursor.fetchall()
            df = pd.DataFrame(data, columns=["smiles", "qed"])
            print(f"Got {len(df)} molecules")
            print("Preparing training data")
            training_data = prepare_training_data(df, num_atoms=max_atoms)
            print("Training the model")
            model.fit(training_data, epochs=epochs, verbose=1)

        molecules = model.inference(molecules_to_generate)

        for i in range(50):
            mol = model.sample_around_molecule(df.loc[i, "smiles"], 2)
            if mol is None:
                continue
            print(Chem.MolToSmiles(mol))
        fault_counter = 0

        for i, molecule in enumerate(molecules):
            if molecule is None:
                fault_counter += 1
                continue
            else:
                qed = QED.qed(molecule)
                smiles = Chem.CanonSmiles(Chem.MolToSmiles(molecule))
                await cursor.execute("INSERT INTO generated_molecules (canonicalsmiles, qed) VALUES (%s, %s)", (smiles, qed))
        print(f"Generated {molecules_to_generate - fault_counter} valid molecules with fault rate {100 * fault_counter / molecules_to_generate}%")

stages: List[Callable[[aiomysql.Connection], Awaitable[None]]] = [
    #insert_qed,
    #prepare_generation_source,
    generate_new,
]

async def main():
    pool = await aiomysql.create_pool(
        host='DESKTOP-HEPGSBI.local',
        port=3306,
        user='bio',
        password='bio',
        db='bio',
        autocommit=True
    )

    async with pool.acquire() as conn:
        for stage in stages:
            await stage(conn)

    pool.close()
    await pool.wait_closed()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())