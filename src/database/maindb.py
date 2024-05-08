import asyncio

from prisma import Prisma


async def main() -> None:
    db = Prisma()
    await db.connect()

    post = await db.identifiedpersondata.create(
        {
            'personName': 'Hello from prisma!'
        }
    )
    

    found = await db.identifiedpersondata.find_first()
    assert found is not None
    print(found)

    await db.disconnect()


if __name__ == '__main__':
    asyncio.run(main())