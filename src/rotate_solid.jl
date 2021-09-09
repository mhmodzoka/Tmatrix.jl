using GLMakie
#input format: scattering cross section magnitude (R), emiss cross section ((x,y) pointlist)

granularity = 20
pointlist = [(0,0,0), (0,5,0)]
for i in 1:2
    push!(pointlist,(rand((0,5)),rand((0,5)),rand((0,5))))
end

function draw_mesh(points, faces)
    scene = mesh(vertices, faces, shading = false)
end

function make_mesh(granularity::Int, pointlist)
    new_pointlist = []
    for point in pointlist
        for slice in 0:granularity-1
            theta = slice*pi/granularity
            point_x = cos(theta)*point[1]
            point_y = point[2]
            point_z = sin(theta)*point[1]
            if new_pointlist == []
                new_pointlist = [point_x, point_y, point_z]
            else
                vcat(new_pointlist, [point_x, point_y, point_z])
            end
        end
    end

    face_list = []
    for u in 0:(granularity-1), v in 0:(granularity-1)
        if face_list == []
            push!(
                face_list,
                (
                    granularity*v + u + 1,
                    mod(granularity*v + mod(u + 1, granularity), granularity*granularity) + 1,
                    mod(granularity*(v + 1) + u, granularity*granularity) + 1,
                ),
            )
        else
            hcat(
                face_list,
                (
                    granularity*v + u + 1,
                    mod(granularity*v + mod(u + 1, granularity), granularity*granularity) + 1,
                    mod(granularity*(v + 1) + u, granularity*granularity) + 1,
                ),
            )
        end
        hcat(
            face_list,
            (
                mod(granularity*v + mod(u + 1, granularity), granularity*granularity) + 1,
                mod(granularity*(v + 1) + mod(u + 1, granularity), granularity*granularity) + 1,
                mod(granularity*(v + 1) + u, granularity*granularity) + 1,
            ),
        )
    end

    draw_mesh(new_pointlist, face_list)
end


make_mesh(20, pointlist)
